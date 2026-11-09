"""Gate and readout timing from the Qblox compile flow (qblox-scheduler).

The twin times circuits from DeviceCal: ``sq_duration_ns`` per qubit and
``cz_duration_ns`` per coupler. This module asks qblox-scheduler instead, so
the twin runs a circuit on the schedule the Qblox stack would play:

1. ``schedule_from_circuit`` turns the lowered native ops
   (``qasm.parse_qasm(...).ops``) into a ``TimeableSchedule``: one ``Rxy``
   per physical pulse, ``CZ`` per cz, a ``Measure`` on the measured qubits
   and, when every element has one, a ``Reset`` first. Each operation waits
   for the last one on each of its qubits (ASAP), the schedule the twin
   builds itself; the measurement waits for every qubit, as the twin's does.
2. ``compile_schedule`` compiles it with a device compilation config (from a
   ``QuantumDevice`` or a ``DeviceCompilationConfig``) and, optionally, a
   Qblox hardware config for a (dummy) Cluster, which also generates the
   Q1ASM and so checks the circuit fits the instruments.
3. ``durations_from_qblox_scheduler`` reads each operation's start and
   duration from the compiled schedule into a ``QbloxTiming``, which
   ``TwinBackend(timing=...)`` simulates on.

``qblox_operation_durations`` and ``apply_qblox_durations`` are the fast
path: time one of each operation per element once and write the results
into a DeviceCal, after which the twin's own ASAP schedule matches Qblox's.

``device_compilation_config`` and ``dummy_cluster_hardware_config`` build a
device config and a Cluster config from a DeviceCal, for trying this without
a lab configuration. Transmon ``QuantumDevice`` files need the element
classes in ``qblox_applications``; a ``DeviceCompilationConfig`` needs only
qblox-scheduler.

Not done: running the compiled Q1ASM in Q1Simulator to catch real-time
buffer underruns. q1simulator 1.4 imports a Qt GUI when loaded, so it needs
Qt bindings even headless.

Written against qblox-scheduler 1.0. qblox-scheduler is imported only inside
these functions: nothing else in the twin needs it.
"""
from __future__ import annotations

import cmath
import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

from .calibration import DeviceCal
from .qasm import Circuit

# Labels the twin gives the schedulables it adds, to find them after compile.
_OP_LABEL = "twin:{index}:{pulse}"
_MEASURE_LABEL = "twin:measure"
_RESET_LABEL = "twin:reset"

# Placeholders for device_compilation_config(): DeviceCal has no readout or
# reset timing yet, so these are illustrative, like the rest of example_4q.
DEFAULT_READOUT_PULSE_NS = 2000.0
DEFAULT_ACQ_DELAY_NS = 100.0
DEFAULT_INTEGRATION_NS = 1900.0
DEFAULT_RESET_NS = 200_000.0

ElementNames = Union[None, Mapping[int, str], Callable[[int], str]]


def _ns(seconds_or_ns: float) -> float:
    # Times to the picosecond: the conversion from seconds and the offsets
    # otherwise leave float dust (2.9e-11 ns) on grid-aligned times.
    return round(seconds_or_ns, 3)


@dataclass
class OpTiming:
    start_ns: float
    duration_ns: float

    @property
    def end_ns(self) -> float:
        return self.start_ns + self.duration_ns


@dataclass
class QbloxTiming:
    """The compiled schedule's times for one circuit, in ns from its start
    (the end of the reset, when there is one).

    ``ops[i]`` times ``circuit.ops[i]``: from the start of its first pulse
    to the end of its last. Virtual ops (no pulses) are ``None``: they take
    no time.
    """
    ops: List[Optional[OpTiming]]
    measure: Optional[OpTiming]
    reset_ns: float
    duration_ns: float          # gates and measurement, without the reset
    compiled_schedule: Any = field(default=None, repr=False, compare=False)
    source = "qblox-scheduler"      # what TwinBackend reports as its timing

    @property
    def gates_end_ns(self) -> float:
        ends = [t.end_ns for t in self.ops if t is not None]
        return max(ends) if ends else 0.0


# --------------------------------------------------------------- configs
def compilation_config(device, hardware=None):
    """A SerialCompilationConfig for ``device`` and, optionally, ``hardware``.

    ``device`` is a QuantumDevice, a DeviceCompilationConfig, a
    SerialCompilationConfig, a dict, or a path to a QuantumDevice YAML/JSON
    file or a DeviceCompilationConfig JSON file. ``hardware`` is a Qblox
    hardware config (model, dict or JSON path); without one the schedule is
    compiled to the device layer only, which is enough for timing. A
    QuantumDevice with a hardware config of its own keeps it unless
    ``hardware`` is given.
    """
    from qblox_scheduler.backends.graph_compilation import (
        DeviceCompilationConfig, SerialCompilationConfig)

    config = None
    if isinstance(device, SerialCompilationConfig):
        config = device
    elif isinstance(device, DeviceCompilationConfig):
        config = SerialCompilationConfig(
            name="qblox-twin", device_compilation_config=device)
    elif hasattr(device, "generate_compilation_config"):
        config = device.generate_compilation_config()
    else:
        device_config = _load_device(device)
        if hasattr(device_config, "generate_compilation_config"):
            config = device_config.generate_compilation_config()
        else:
            config = SerialCompilationConfig(
                name="qblox-twin", device_compilation_config=device_config)
    if hardware is not None:
        config = config.model_copy(
            update={"hardware_compilation_config": _load_hardware(hardware)})
    return config


def _load_device(device):
    from qblox_scheduler import QuantumDevice
    from qblox_scheduler.backends.graph_compilation import DeviceCompilationConfig

    if isinstance(device, (str, Path)):
        path = Path(device)
        if path.suffix.lower() in (".yaml", ".yml"):
            return QuantumDevice.from_yaml_file(str(path))
        data = json.loads(path.read_text())
        if _is_device_compilation_config(data):
            return DeviceCompilationConfig.model_validate(data)
        return QuantumDevice.from_json_file(str(path))
    if isinstance(device, dict):
        if _is_device_compilation_config(device):
            return DeviceCompilationConfig.model_validate(device)
        return QuantumDevice.from_dict(device)
    raise TypeError(f"cannot make a device compilation config from {type(device).__name__}")


def _is_device_compilation_config(data: dict) -> bool:
    # A DeviceCompilationConfig maps elements to operation configs with a
    # factory_func; a QuantumDevice maps them to element parameters.
    if "clocks" in data or "compilation_passes" in data:
        return True
    for element in (data.get("elements") or {}).values():
        for operation in (element or {}).values():
            if isinstance(operation, dict) and "factory_func" in operation:
                return True
    return False


def _load_hardware(hardware):
    from qblox_scheduler.backends.qblox_backend import QbloxHardwareCompilationConfig

    if isinstance(hardware, QbloxHardwareCompilationConfig):
        return hardware
    if isinstance(hardware, (str, Path)):
        hardware = json.loads(Path(hardware).read_text())
    if isinstance(hardware, dict):
        return QbloxHardwareCompilationConfig.model_validate(hardware)
    return hardware


def _element_name(names: ElementNames, qubit: int) -> str:
    if names is None:
        return f"q{qubit}"
    if callable(names):
        return names(qubit)
    return names[qubit]


# -------------------------------------------------------------- schedule
def _rxy_angles(unitary, pulses: int) -> List[Tuple[float, float]]:
    """(theta, phi) in degrees for each Rxy pulse of a 1q op.

    One pulse: the ZYZ angles of the unitary give U ~ Rz(a) Rxy(theta, 90 - l),
    so the pulse is exact and only a virtual Z follows. Two pulses (the
    twin's u3): two X90s. Virtual Z frame changes take no time and are left
    out, so the schedule times the circuit; it is not a phase-exact program.
    """
    if pulses == 2:
        return [(90.0, 0.0), (90.0, 0.0)]
    u = np.asarray(unitary, dtype=complex)
    theta = 2 * math.atan2(abs(u[1, 0]), abs(u[0, 0]))
    lam = 0.0
    if abs(u[0, 0]) > 1e-9 and abs(u[1, 0]) > 1e-9:
        plus = cmath.phase(u[1, 1]) - cmath.phase(u[0, 0])       # phi + lambda
        minus = cmath.phase(u[1, 0]) - cmath.phase(-u[0, 1])     # phi - lambda
        lam = (plus - minus) / 2
    elif abs(u[1, 0]) > 1e-9:
        lam = -(cmath.phase(u[1, 0]) - cmath.phase(-u[0, 1])) / 2
    # Halving the phase difference leaves phi known only up to 180 degrees,
    # which would reverse the rotation: keep the one that leaves a Z.
    phi = math.pi / 2 - lam
    for candidate in (phi, phi + math.pi):
        rest = u @ rxy_matrix(theta, candidate).conj().T
        if abs(rest[0, 1]) < 1e-6 and abs(rest[1, 0]) < 1e-6:
            phi = candidate
            break
    return [(math.degrees(theta), math.degrees(phi) % 360.0)]


def rxy_matrix(theta: float, phi: float):
    """Rxy(theta, phi) = exp(-i theta/2 (cos phi X + sin phi Y)), radians."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * cmath.exp(-1j * phi) * s],
                     [-1j * cmath.exp(1j * phi) * s, c]])


def _has_reset(config, elements) -> bool:
    device = config.device_compilation_config
    if device is None:
        return False
    return all("reset" in (device.elements.get(e) or {}) for e in elements)


def schedule_from_circuit(circuit: Circuit, element_names: ElementNames = None,
                          reset: Optional[bool] = True, name: str = "qblox-twin"):
    """The circuit's native ops as an ASAP TimeableSchedule (see module doc)."""
    from qblox_scheduler.operations.gate_library import CZ, Measure, Reset, Rxy
    from qblox_scheduler.schedules.schedule import TimeableSchedule

    schedule = TimeableSchedule(name)
    last: Dict[int, Any] = {}
    start = None
    used = sorted({q for op in circuit.ops for q in op.qubits} |
                  set(circuit.measures.values()))
    if reset:
        start = schedule.add(Reset(*[_element_name(element_names, q) for q in used]),
                             label=_RESET_LABEL)
        last = {q: start for q in used}

    def add(operation, qubits, label):
        nonlocal start
        refs = []
        for q in qubits:
            if q in last and last[q] not in refs:
                refs.append(last[q])
        if refs:
            schedulable = schedule.add(operation, ref_op=refs[0], ref_pt="end", label=label)
            for ref in refs[1:]:
                schedulable.add_timing_constraint(ref_schedulable=ref, ref_pt="end")
        elif start is None:
            schedulable = schedule.add(operation, label=label)
        else:
            # A qubit's first op starts with the schedule's first one.
            schedulable = schedule.add(operation, ref_op=start, ref_pt="start", label=label)
        if start is None:
            start = schedulable
        for q in qubits:
            last[q] = schedulable

    for index, op in enumerate(circuit.ops):
        if op.kind == "sq":
            element = _element_name(element_names, op.qubits[0])
            for pulse, (theta, phi) in enumerate(_rxy_angles(op.unitary, op.pulses)
                                                 if op.pulses else []):
                add(Rxy(theta=theta, phi=phi, qubit=element), op.qubits,
                    _OP_LABEL.format(index=index, pulse=pulse))
        else:
            a, b = op.qubits
            add(CZ(_element_name(element_names, a), _element_name(element_names, b)),
                op.qubits, _OP_LABEL.format(index=index, pulse=0))
    measured = [circuit.measures[c] for c in sorted(circuit.measures)]
    if measured:
        add(Measure(*[_element_name(element_names, q) for q in measured]),
            used, _MEASURE_LABEL)
    return schedule


def compile_schedule(schedule, config):
    from qblox_scheduler.backends.graph_compilation import SerialCompiler

    return SerialCompiler(name="qblox-twin").compile(schedule=schedule, config=config)


def _labelled_times(compiled, prefix: str = "twin:") -> Dict[str, OpTiming]:
    """Start and duration (ns, from the schedule's start) of every
    schedulable whose label starts with ``prefix``, at any depth: compiling
    wraps the schedule in a repetition loop and gates in subschedules."""
    from qblox_scheduler.operations.control_flow_library import ControlFlowOperation
    from qblox_scheduler.schedules.schedule import TimeableScheduleBase

    found: Dict[str, OpTiming] = {}

    def walk(schedule, offset):
        for label, schedulable in schedule.schedulables.items():
            operation = schedule.operations[schedulable["operation_id"]]
            start = offset + schedulable["abs_time"]
            if str(label).startswith(prefix):
                found[str(label)] = OpTiming(_ns(start * 1e9), _ns(operation.duration * 1e9))
            elif isinstance(operation, ControlFlowOperation):
                walk(operation.body, start)
            elif isinstance(operation, TimeableScheduleBase):
                walk(operation, start)

    walk(compiled, 0.0)
    return found


def durations_from_qblox_scheduler(circuit: Circuit, quantum_device, hardware_config=None,
                                   element_names: ElementNames = None,
                                   reset: Optional[bool] = None) -> QbloxTiming:
    """Time ``circuit`` with qblox-scheduler (see module doc).

    ``quantum_device`` is anything ``compilation_config`` takes, including a
    config it returned (reuse one to skip reloading files per circuit).
    ``element_names`` maps the twin's qubit indices to device elements
    (default ``q<index>``). ``reset=None`` adds a Reset when every element
    has one configured.
    """
    config = compilation_config(quantum_device, hardware_config)
    if reset is None:
        used = {q for op in circuit.ops for q in op.qubits} | set(circuit.measures.values())
        reset = _has_reset(config, [_element_name(element_names, q) for q in used])
    compiled = compile_schedule(
        schedule_from_circuit(circuit, element_names, reset=reset), config)
    times = _labelled_times(compiled)

    reset_time = times.get(_RESET_LABEL)
    pulses = [t for label, t in times.items() if label not in (_RESET_LABEL, _MEASURE_LABEL)]
    if reset_time is not None:
        origin = reset_time.end_ns
    else:
        starts = [t.start_ns for t in pulses]
        if _MEASURE_LABEL in times:
            starts.append(times[_MEASURE_LABEL].start_ns)
        origin = min(starts) if starts else 0.0

    ops: List[Optional[OpTiming]] = []
    for index, op in enumerate(circuit.ops):
        parts = [times[_OP_LABEL.format(index=index, pulse=p)]
                 for p in range(op.pulses if op.kind == "sq" else 1)]
        if not parts:
            ops.append(None)
            continue
        start = _ns(parts[0].start_ns - origin)
        ops.append(OpTiming(start, _ns(parts[-1].end_ns - origin - start)))
    measure = None
    if _MEASURE_LABEL in times:
        m = times[_MEASURE_LABEL]
        measure = OpTiming(_ns(m.start_ns - origin), m.duration_ns)
    ends = [t.end_ns for t in ops if t is not None]
    if measure is not None:
        ends.append(measure.end_ns)
    return QbloxTiming(
        ops=ops, measure=measure,
        reset_ns=reset_time.duration_ns if reset_time is not None else 0.0,
        duration_ns=_ns(max(ends)) if ends else 0.0,
        compiled_schedule=compiled)


# ------------------------------------------------- per-element durations
def qblox_operation_durations(device_cal: DeviceCal, quantum_device, hardware_config=None,
                              element_names: ElementNames = None,
                              reset: Optional[bool] = None) -> Dict[tuple, float]:
    """Duration (ns) of each operation on each of the twin's elements:
    ``("rxy", q)``, ``("cz", a, b)``, ``("measure", q)`` and, when configured,
    ``("reset", q)``. One schedule, compiled once."""
    from qblox_scheduler.operations.gate_library import CZ, Measure, Reset, Rxy
    from qblox_scheduler.schedules.schedule import TimeableSchedule

    config = compilation_config(quantum_device, hardware_config)
    names = {q: _element_name(element_names, q) for q in device_cal.qubits}
    if reset is None:
        reset = _has_reset(config, list(names.values()))
    schedule = TimeableSchedule("qblox-twin durations")
    for q, element in names.items():
        schedule.add(Rxy(theta=180.0, phi=0.0, qubit=element), label=f"probe:rxy:{q}")
        schedule.add(Measure(element), label=f"probe:measure:{q}")
        if reset:
            schedule.add(Reset(element), label=f"probe:reset:{q}")
    for coupler in device_cal.couplers:
        a, b = coupler.qubits
        schedule.add(CZ(names[a], names[b]), label=f"probe:cz:{a}:{b}")
    times = _labelled_times(compile_schedule(schedule, config), prefix="probe:")

    durations: Dict[tuple, float] = {}
    for label, timing in times.items():
        kind, *qubits = label.split(":")[1:]
        durations[(kind, *(int(q) for q in qubits))] = timing.duration_ns
    return durations


def apply_qblox_durations(device_cal: DeviceCal, durations: Mapping[tuple, float]) -> DeviceCal:
    """A copy of ``device_cal`` with the gate durations from ``durations``
    (as ``qblox_operation_durations`` returns them). Readout and reset have
    no field in DeviceCal yet, so they are not applied."""
    updated = copy.deepcopy(device_cal)
    for q, cal in updated.qubits.items():
        if ("rxy", q) in durations:
            cal.sq_duration_ns = durations[("rxy", q)]
    for coupler in updated.couplers:
        a, b = coupler.qubits
        for key in (("cz", a, b), ("cz", b, a)):
            if key in durations:
                coupler.cz_duration_ns = durations[key]
                break
    return updated


# ------------------------------------------- configs from the twin itself
def _twin_measurement(port, clock, pulse_amp, pulse_duration, acq_delay,
                      integration_time, acq_label=None, **_ignored):
    """Square readout pulse with an IQ integration acq_delay after its start."""
    from qblox_scheduler.operations.acquisition_library import IQIntegration
    from qblox_scheduler.operations.pulse_library import SquarePulse
    from qblox_scheduler.schedules.schedule import TimeableSchedule

    schedule = TimeableSchedule("qblox-twin measurement")
    schedule.add(SquarePulse(amplitude=pulse_amp, duration=pulse_duration,
                             port=port, clock=clock))
    schedule.add(IQIntegration(port=port, clock=clock, duration=integration_time,
                               acq_label=0 if acq_label is None else acq_label),
                 ref_pt="start", rel_time=acq_delay)
    return schedule


def _twin_reset(duration, **_ignored):
    from qblox_scheduler.operations.pulse_library import IdlePulse

    return IdlePulse(duration=duration)


def device_compilation_config(device_cal: DeviceCal, element_names: ElementNames = None,
                              readout_pulse_ns: float = DEFAULT_READOUT_PULSE_NS,
                              acq_delay_ns: float = DEFAULT_ACQ_DELAY_NS,
                              integration_ns: float = DEFAULT_INTEGRATION_NS,
                              reset_ns: float = DEFAULT_RESET_NS):
    """A DeviceCompilationConfig for the twin's device: DRAG Rxy of
    ``sq_duration_ns``, a composite square CZ of ``cz_duration_ns`` on each
    coupler's flux port, a square readout with IQ integration, and a passive
    reset. Ports and clocks follow qblox-scheduler's transmon convention
    (``q0:mw``, ``q0:fl``, ``q0:res``, ``q0.01``, ``q0.ro``); frequencies and
    amplitudes are placeholders, since timing does not depend on them.
    """
    from qblox_scheduler.backends.graph_compilation import (
        DeviceCompilationConfig, OperationCompilationConfig)
    from qblox_scheduler.operations.pulse_factories import (
        composite_square_pulse, rxy_drag_pulse)

    elements, clocks, edges = {}, {}, {}
    for q, cal in sorted(device_cal.qubits.items()):
        e = _element_name(element_names, q)
        clocks[f"{e}.01"] = 5.0e9 + 0.1e9 * q
        clocks[f"{e}.ro"] = 7.0e9 + 0.05e9 * q
        elements[e] = {
            "Rxy": OperationCompilationConfig(
                factory_func=rxy_drag_pulse,
                factory_kwargs={"amp180": 0.1, "beta": 0.0, "port": f"{e}:mw",
                                "clock": f"{e}.01", "duration": cal.sq_duration_ns * 1e-9},
                gate_info_factory_kwargs=["theta", "phi"]),
            "measure": OperationCompilationConfig(
                factory_func=_twin_measurement,
                factory_kwargs={"port": f"{e}:res", "clock": f"{e}.ro", "pulse_amp": 0.05,
                                "pulse_duration": readout_pulse_ns * 1e-9,
                                "acq_delay": acq_delay_ns * 1e-9,
                                "integration_time": integration_ns * 1e-9}),
            "reset": OperationCompilationConfig(
                factory_func=_twin_reset, factory_kwargs={"duration": reset_ns * 1e-9}),
        }
    for coupler in device_cal.couplers:
        a, b = (_element_name(element_names, q) for q in coupler.qubits)
        edges[f"{a}_{b}"] = {
            "CZ": OperationCompilationConfig(
                factory_func=composite_square_pulse,
                factory_kwargs={"square_amp": 0.1,
                                "square_duration": coupler.cz_duration_ns * 1e-9,
                                "square_port": f"{a}:fl", "square_clock": "cl0.baseband",
                                "virt_z_parent_qubit_phase": 0.0,
                                "virt_z_parent_qubit_clock": f"{a}.01",
                                "virt_z_child_qubit_phase": 0.0,
                                "virt_z_child_qubit_clock": f"{b}.01"})}
    return DeviceCompilationConfig(elements=elements, clocks=clocks, edges=edges)


def dummy_cluster_hardware_config(device_cal: DeviceCal,
                                  element_names: ElementNames = None) -> dict:
    """A QbloxHardwareCompilationConfig (as a dict) for a dummy Cluster that
    can play ``device_compilation_config``: a QCM-RF per two drive lines, a
    QCM per four flux lines, and one QRM-RF reading out every resonator."""
    names = [_element_name(element_names, q) for q in sorted(device_cal.qubits)]
    modules: Dict[str, dict] = {}
    graph: List[list] = []
    frequencies: Dict[str, dict] = {}
    slot = 1
    for first in range(0, len(names), 2):
        modules[str(slot)] = {"instrument_type": "QCM_RF"}
        for output, e in enumerate(names[first:first + 2]):
            graph.append([f"cluster0.module{slot}.complex_output_{output}", f"{e}:mw"])
            frequencies[f"{e}:mw-{e}.01"] = {"interm_freq": 50e6}
        slot += 1
    for first in range(0, len(names), 4):
        modules[str(slot)] = {"instrument_type": "QCM"}
        for output, e in enumerate(names[first:first + 4]):
            graph.append([f"cluster0.module{slot}.real_output_{output}", f"{e}:fl"])
        slot += 1
    modules[str(slot)] = {"instrument_type": "QRM_RF"}
    graph.append([f"cluster0.module{slot}.complex_output_0", [f"{e}:res" for e in names]])
    for e in names:
        frequencies[f"{e}:res-{e}.ro"] = {"lo_freq": 6.9e9}
    return {
        "config_type": "QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {"instrument_type": "Cluster", "ref": "internal", "modules": modules}},
        "hardware_options": {"modulation_frequencies": frequencies},
        "connectivity": {"graph": graph},
    }
