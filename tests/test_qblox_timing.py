"""The twin's Qblox timing (services/svc_qblox_twin_qpm/qfw_qblox_twin/qblox_timing.py).

Needs numpy and qblox-scheduler (tested with 1.0.1); skips without them.
The device configs come from the twin's own calibration, so no
qblox_applications element classes are needed.
"""
import copy
import math
import pathlib
import sys
import warnings

import pytest

np = pytest.importorskip("numpy")
if not hasattr(np, "random"):
    pytest.skip("numpy is the tests/mock stub here; run this file on its own",
                allow_module_level=True)
pytest.importorskip("qblox_scheduler")

SERVICE_DIR = (pathlib.Path(__file__).resolve().parents[1] /
               "services" / "svc_qblox_twin_qpm")
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from qfw_qblox_twin import DeviceCal, TwinBackend, hellinger_fidelity  # noqa: E402
from qfw_qblox_twin import qblox_timing as qt  # noqa: E402
from qfw_qblox_twin.qasm import _ONE_Q, parse_qasm  # noqa: E402
from qfw_qblox_twin.qpm_service import QbloxTwinQPM  # noqa: E402

HDR = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
GHZ4 = (HDR + "qreg q[4];\ncreg c[4];\nh q[0];\ncx q[0],q[1];\ncx q[1],q[2];\n"
        "cx q[2],q[3];\nmeasure q -> c;\n")


@pytest.fixture(autouse=True)
def _quiet():
    # qblox-scheduler warns about unset hardware options on every compile.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="module")
def cal():
    return DeviceCal.example_4q()


@pytest.fixture(scope="module")
def dcc(cal):
    return qt.device_compilation_config(cal)


def _slower(cal, sq_ns=None, cz_scale=1.0):
    slow = copy.deepcopy(cal)
    for q in slow.qubits.values():
        if sq_ns is not None:
            q.sq_duration_ns = sq_ns
    for c in slow.couplers:
        c.cz_duration_ns *= cz_scale
    return slow


def test_operation_durations_come_from_the_device_config(cal, dcc):
    durations = qt.qblox_operation_durations(cal, dcc)

    assert {k: v for k, v in durations.items() if k[0] == "rxy"} == {
        ("rxy", q): 40.0 for q in range(4)}
    assert durations[("cz", 0, 1)] == 80.0
    assert durations[("cz", 1, 2)] == 100.0
    assert durations[("cz", 2, 3)] == 120.0
    assert durations[("measure", 2)] == qt.DEFAULT_READOUT_PULSE_NS
    assert durations[("reset", 3)] == qt.DEFAULT_RESET_NS


def test_apply_qblox_durations_rewrites_gate_times_only(cal):
    qblox = qt.device_compilation_config(_slower(cal, sq_ns=48.0, cz_scale=2.0))

    updated = qt.apply_qblox_durations(cal, qt.qblox_operation_durations(cal, qblox))

    assert [q.sq_duration_ns for q in updated.qubits.values()] == [48.0] * 4
    assert [c.cz_duration_ns for c in updated.couplers] == [160.0, 200.0, 240.0]
    assert [q.t1_us for q in updated.qubits.values()] == [q.t1_us for q in cal.qubits.values()]
    assert cal.qubits[0].sq_duration_ns == 40.0          # the original is untouched


def test_circuit_timing_is_the_twins_asap_schedule(cal, dcc):
    timing = qt.durations_from_qblox_scheduler(parse_qasm(GHZ4), dcc)
    _, meta = TwinBackend(cal).exact_probabilities(GHZ4)

    # GHZ-4 lowers to h, then H CZ H per cx: the H on each target can run
    # at once, each CZ waits for both of its qubits.
    assert [t.start_ns for t in timing.ops] == [0, 0, 40, 120, 0, 160, 260, 0, 300, 420]
    assert [t.duration_ns for t in timing.ops] == [40, 40, 80, 40, 40, 100, 40, 40, 120, 40]
    assert timing.measure.start_ns == meta["duration_ns"] == 460.0
    assert timing.measure.duration_ns == qt.DEFAULT_READOUT_PULSE_NS
    assert timing.reset_ns == qt.DEFAULT_RESET_NS
    assert timing.duration_ns == 460.0 + qt.DEFAULT_READOUT_PULSE_NS
    assert timing.gates_end_ns == 460.0


def test_twin_on_qblox_timing_matches_nominal_when_durations_agree(cal, dcc):
    nominal, _ = TwinBackend(cal).exact_probabilities(GHZ4)
    qblox, meta = TwinBackend(
        cal, timing=lambda c: qt.durations_from_qblox_scheduler(c, dcc)
    ).exact_probabilities(GHZ4)

    assert meta["timing"] == "qblox-scheduler"
    assert set(qblox) == set(nominal)
    assert max(abs(qblox[k] - nominal[k]) for k in nominal) < 1e-12


def test_slower_qblox_gates_cost_fidelity(cal):
    slow = qt.device_compilation_config(_slower(cal, cz_scale=10.0))
    ideal, _ = TwinBackend(cal.ideal_copy()).exact_probabilities(GHZ4)
    nominal, nominal_meta = TwinBackend(cal).exact_probabilities(GHZ4)
    on_qblox, meta = TwinBackend(
        cal, timing=lambda c: qt.durations_from_qblox_scheduler(c, slow)
    ).exact_probabilities(GHZ4)

    assert meta["duration_ns"] == 460.0 + 9 * (80 + 100 + 120)
    assert hellinger_fidelity(on_qblox, ideal) < hellinger_fidelity(nominal, ideal)


def test_hardware_compile_on_a_dummy_cluster(cal, dcc):
    circuit = parse_qasm(GHZ4)
    hardware = qt.dummy_cluster_hardware_config(cal)

    on_cluster = qt.durations_from_qblox_scheduler(circuit, dcc, hardware)
    device_only = qt.durations_from_qblox_scheduler(circuit, dcc)

    assert on_cluster.ops == device_only.ops
    assert on_cluster.measure == device_only.measure
    instructions = on_cluster.compiled_schedule.compiled_instructions["cluster0"]
    modules = [k for k in instructions if k.startswith("cluster0_module")]
    assert len(modules) == 4                 # 2 QCM-RF, 1 QCM, 1 QRM-RF
    programs = []

    def collect(node):
        # module -> "sequencers" -> sequencer settings, whose .sequence
        # holds the Q1ASM program
        sequence = getattr(node, "sequence", None)
        if isinstance(sequence, dict) and isinstance(sequence.get("program"), str):
            programs.append(sequence["program"])
        elif isinstance(node, dict):
            for value in node.values():
                collect(value)
    collect(instructions)
    assert any("play" in p for p in programs)          # Q1ASM was generated


def test_reversed_cz_virtual_z_and_two_pulse_gates(dcc):
    circuit = parse_qasm(HDR + "qreg q[2];\ncreg c[2];\nx q[1];\ncx q[1],q[0];\n"
                         "u3(0.1,0.2,0.3) q[0];\nrz(0.5) q[1];\nmeasure q -> c;\n")

    timing = qt.durations_from_qblox_scheduler(circuit, dcc)

    # x, H(t), CZ(1,0) on edge q0_q1, H(t), u3 = two pulses, rz is virtual
    assert [(t.start_ns, t.duration_ns) if t else None for t in timing.ops] == [
        (0, 40), (0, 40), (40, 80), (120, 40), (160, 80), None]


def test_element_names_map_twin_qubits(cal):
    names = {q: f"transmon_{q}" for q in range(4)}
    renamed = qt.device_compilation_config(cal, element_names=names)

    timing = qt.durations_from_qblox_scheduler(
        parse_qasm(GHZ4), renamed, element_names=names)
    assert timing.measure.start_ns == 460.0
    with pytest.raises(Exception, match="q0"):
        qt.durations_from_qblox_scheduler(parse_qasm(GHZ4), renamed)


def test_device_config_file_round_trips(cal, dcc, tmp_path):
    path = tmp_path / "device_compilation_config.json"
    path.write_text(dcc.model_dump_json())
    hardware = tmp_path / "hardware.json"
    hardware.write_text(__import__("json").dumps(qt.dummy_cluster_hardware_config(cal)))

    config = qt.compilation_config(str(path), str(hardware))

    assert config.hardware_compilation_config is not None
    assert qt.qblox_operation_durations(cal, config) == qt.qblox_operation_durations(cal, dcc)


@pytest.mark.parametrize("name", ["x", "y", "h", "sx", "sxdg"])
def test_single_pulse_is_exact_up_to_a_virtual_z(name):
    n_params, factory, pulses = _ONE_Q[name]
    _assert_pulse_then_z(factory(), pulses)


def test_random_rotations_are_exact_up_to_a_virtual_z():
    rng = np.random.default_rng(5)
    for _ in range(200):
        theta, phi, lam = rng.uniform(-math.pi, math.pi, 3)
        _assert_pulse_then_z(_ONE_Q["u3"][1](abs(theta), phi, lam), 1)
        _assert_pulse_then_z(_ONE_Q["rx"][1](theta), 1)
        _assert_pulse_then_z(_ONE_Q["ry"][1](theta), 1)


def _assert_pulse_then_z(unitary, pulses):
    (theta, phi), = qt._rxy_angles(unitary, pulses)
    rest = unitary @ qt.rxy_matrix(math.radians(theta), math.radians(phi)).conj().T
    assert abs(rest[0, 1]) < 1e-6 and abs(rest[1, 0]) < 1e-6, (theta, phi)


def test_qpm_facade_takes_its_timing_from_qblox(cal, tmp_path):
    path = tmp_path / "device.json"
    path.write_text(qt.device_compilation_config(_slower(cal, sq_ns=48.0)).model_dump_json())

    once = QbloxTwinQPM(qblox_device=str(path), seed=1)
    per_circuit = QbloxTwinQPM(qblox_device=str(path), seed=1, qblox_per_circuit=True)
    cid = per_circuit.create_circuit(GHZ4, 200)
    counts = per_circuit.run(cid, sync=True)

    assert QbloxTwinQPM().capabilities()["timing"] == "nominal"
    assert once.capabilities()["timing"] == "qblox-durations"
    assert once.device.qubits[0].sq_duration_ns == 48.0
    assert once.qblox_durations[("measure", 0)] == qt.DEFAULT_READOUT_PULSE_NS
    assert per_circuit.capabilities()["timing"] == "qblox-circuit"
    assert sum(counts.values()) == 200
    assert per_circuit.metadata(cid)["timing"] == "qblox-scheduler"
    assert per_circuit.metadata(cid)["duration_ns"] == 460.0 + 4 * 8
    once.shutdown()
    per_circuit.shutdown()
