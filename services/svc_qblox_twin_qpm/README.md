# svc_qblox_twin_qpm: Qblox transmon digital twin as a QFw QPM

A QPM service whose provider is a digital twin of a Qblox-controlled transmon
chip: OpenQASM 2.0 in, noisy counts out. It is laid out like
`svc_nwqsim_qpm` (`__init__.py`, `svc_qpm.py`, `svc_qrc.py`) and wraps
`QbloxTwinQPM` from the vendored `qfw_qblox_twin` package.

```
svc_qblox_twin_qpm/
  __init__.py          svc_info, service_classes, initialize/uninitialize
  svc_qpm.py           QPM(UTIL_QPM): query, device profile, calibration telemetry
  svc_qrc.py           QRC(UTIL_QRC): runs circuits on QbloxTwinQPM, in-process
  device.example.json  calibration template (same numbers as DeviceCal.example_4q())
  qfw_qblox_twin/      the twin: parser, noise model, engines, QbloxTwinQPM
```

## How it differs from svc_nwqsim_qpm

| | nwqsim | qblox-twin |
|---|---|---|
| Execution | QASM file + `circuit_runner.nwqsim` under MPI, polled by UTIL_QRC runners | `QbloxTwinQPM` in the QPM process; no launcher, no MPI |
| QRC | `form_cmd` + `parse_result` | `sync_run` / `async_run` / `cancel` / `shutdown`; keeps UTIL_QRC's completion queue (`read_cq`, `peak_cq`, event push) and result shape |
| Type / capability | SIMULATOR / STATEVECTOR | SIMULATOR / SUPERCONDUCTING, provider `qblox-twin` |
| Slots | 8 per host | 1 per host (a device runs one circuit at a time; QFw's scheduler holds the queue) |
| Device telemetry | `metadata_supported: False` | publishes coupling map, native gates, per-gate error and duration, T1/T2 |
| Cancel | not supported | queued runs cancel; a run in progress finishes (`CANCEL_PENDING`) |

Because the QPM publishes its calibration through `api_qpm_telemetry`, a
`QFwBackend(provider="qblox-twin")` builds a Qiskit `Target` from it
(`backends/qfw_qiskit/qfw_target.py`): native gates `rz` (virtual, free),
`rx`/`ry` (one pulse), `cz` on the coupled pairs, and `measure`, with the
twin's errors and durations. Transpiling against that target makes Qiskit's
noise-aware layout prefer the better qubits; for the example device a Bell
pair lands on (0, 1), not (2, 3).

## Configuration

The service is listed as `qblox-twin` in
`share/qfw/config/services/{local,site}-services.yaml` (internal launch, no
environment modules, no executables). The twin reads these from the
environment of the QPM process:

| Variable | Meaning | Default |
|---|---|---|
| `QFW_QBLOX_TWIN_DEVICE` | calibration JSON (copy `device.example.json`) | `DeviceCal.example_4q()` |
| `QFW_QBLOX_TWIN_ENGINE` | `numpy` or `qutip` | `numpy` |
| `QFW_QBLOX_TWIN_SEED` | integer seed for shot sampling | unseeded |
| `device-id` in the services file (`QFW_QPU_DEVICE_ID`) | id the QPM serves the device under | `qblox-twin-<calibration name>` |
| `QFW_QBLOX_TWIN_QBLOX_DEVICE` | Qblox `QuantumDevice` (YAML/JSON) or `DeviceCompilationConfig` (JSON) to take gate timing from | unset: durations from the calibration JSON |
| `QFW_QBLOX_TWIN_QBLOX_HARDWARE` | Qblox hardware config JSON, so compiles also generate and check Q1ASM | unset: device-layer compile |
| `QFW_QBLOX_TWIN_TIMING` | `durations`: gate durations from Qblox once at start; `circuit`: also compile every circuit and simulate on its schedule | `durations` |
| `QFW_QBLOX_TWIN_QBLOX_ELEMENTS` | device element for each twin qubit, comma separated | `q0,q1,...` |

The calibration set id the QPM publishes is the calibration's name plus a
hash of its numbers, so it changes whenever the JSON does, and when Qblox
timing replaces its durations. The `timing` query property says which
timing is in use (`nominal`, `qblox-durations` or `qblox-circuit`).

## Timing from Qblox

`qfw_qblox_twin/qblox_timing.py` builds a qblox-scheduler `TimeableSchedule`
from the twin's native ops (one `Rxy` per pulse, `CZ`, a `Measure` after
every qubit's last gate, a `Reset` first when configured), with each
operation waiting on the last one on each of its qubits: the ASAP schedule
the twin uses itself. It compiles that with your device config (and
hardware config, if given) and reads each operation's start and duration
back. With the twin's own durations the two schedules agree to the
picosecond; with yours, the twin simulates decoherence on Qblox's times.

```python
from qfw_qblox_twin import DeviceCal, TwinBackend, qblox_timing as qt
cal = DeviceCal.load("device.json")
config = qt.compilation_config("quantum_device.yaml", "hardware_config.json")
cal = qt.apply_qblox_durations(cal, qt.qblox_operation_durations(cal, config))  # once
twin = TwinBackend(cal, timing=lambda c: qt.durations_from_qblox_scheduler(c, config))
```

A device-layer compile of a small circuit takes milliseconds; with a
hardware config it is about half a second, so `circuit` mode is for
studying schedules rather than throughput. Transmon `QuantumDevice` files
need the element classes from `qblox_applications`;
`qt.device_compilation_config(cal)` and `qt.dummy_cluster_hardware_config(cal)`
build configs from a twin calibration when you have neither. Requires
qblox-scheduler (tested with 1.0.1), which the rest of the twin never imports.

## Running on a live QFw

Nothing here has run on a live QFw yet; every check so far stubbed the DEFw
daemons. These are the steps for a first run.

1. **Clone this fork with its submodules.** DEFw comes from
   `delgadoandrea21/DEFw` (its `master` is the commit QFw pins); the QHW
   packages come from openQSE, because `.gitmodules` on this branch points
   them there (see the note at its top).

   ```bash
   git clone --recursive -b qblox-twin-qpm https://github.com/delgadoandrea21/QFw.git
   ```

   In the QFw-SLURM-Cluster container, do this in the QFw source tree
   (`/workspace/qfw-container-base/QFw`) instead, with
   `git submodule sync && git submodule update --init --recursive` after
   checking out the branch.

2. **Install and activate**, with QFw's standard Python environment
   (Python 3.11 or later). The twin needs nothing beyond it: numpy is already a
   QFw requirement.

   ```bash
   ./setup/qfw_install.sh --prefix <prefix> --with-defw
   source <prefix>/bin/qfw-activate
   ```

3. **Run the GHZ example.** It starts only the `qblox-twin` service, reserves
   it, runs and tears down. On a Slurm cluster, run it inside an allocation as
   for the other examples. The `QFW_QBLOX_TWIN_*` settings are read from the
   environment of the QPM process.

   ```bash
   export QFW_QBLOX_TWIN_SEED=11                     # optional
   export QFW_QBLOX_TWIN_DEVICE=/path/device.json    # optional
   cd "$QFW_SHARE_DIR/examples"
   ./qfw_ghz.sh qiskit 3 qblox-twin 1
   ```

   With the example calibration, about 8% of the GHZ-3 shots land outside
   000 and 111.

4. **Long-running site QPM** (once step 3 works). `qblox-twin` is in the
   packaged site manifest (ports 8790 and 8791). Start it on its host after
   the directory service reports ready:

   ```bash
   qfw-qpm-svc start --scope site --run-dir /var/lib/qfw/qpm/qblox-twin \
     --site-config "$QFW_SITE_CONFIG" --service-id qblox-twin
   ```

   Under systemd, put the `QFW_QBLOX_TWIN_*` settings in
   `/etc/openqse/qfw/services/qpm/qblox-twin.env` and enable
   `qfw-qpm@qblox-twin`. Run examples with
   `--service-mode site --backend qblox-twin`. The virtual cluster's
   `qfw-site-services start` starts a fixed set of services that does not
   include the twin.

From Python, with a reservation as in `examples/tests/test_qiskit_ghz.py`:

```python
from qfw_qiskit import QFwBackend
backend = QFwBackend(provider="qblox-twin")
job = backend.run(backend.compile(circuit), shots=4096, reservation_id=...)
```

`backend.run(circuit)` without `compile` also works: the twin lowers
`h`, `cx`, `swap`, `u3` and the other gates in `capabilities()["instructions"]`
itself. Compiling first is what applies the calibration-aware layout.

## Qblox packages

The twin needs no Qblox packages with its default timing. qblox-scheduler is
only needed to take gate timing from your Qblox configuration, and it **cannot
be installed into QFw's standard environment**: QFw's IQM client stack pins
`ruamel.yaml==0.17.32` (through `iqm-exa-common`), and qblox-scheduler 1.x
needs `ruamel.yaml>=0.18`. A resolver given both either fails or quietly picks
the empty `qblox-scheduler==0.0.0` release. There are three ways around it.

1. **Bake the durations into the calibration file (recommended).** Run
   `bake_qblox_durations.py` (in this directory) once, wherever qblox-scheduler
   is installed (your lab's Qblox environment works), and deploy the file it
   writes. The QPM then uses Qblox's gate durations from the standard QFw
   environment, with no Qblox packages:

   ```bash
   python bake_qblox_durations.py --device quantum_device.yaml \
     --hardware hardware_config.json --calibration device.json \
     --output device.qblox.json
   export QFW_QBLOX_TWIN_DEVICE=/path/device.qblox.json   # on the QPM host
   ```

   It replaces each qubit's `sq_duration_ns` and each coupler's
   `cz_duration_ns`, keeps every other number, and records the Measure and
   Reset durations under `qblox_timing` in the file (the twin does not use
   them yet). `--elements` maps twin qubits to your element names;
   `--dry-run` prints the table without writing. The script needs only this
   directory, numpy and qblox-scheduler, not a QFw install.

2. **Live Qblox timing in the QPM** (`QFW_QBLOX_TWIN_QBLOX_DEVICE`, and
   `QFW_QBLOX_TWIN_TIMING=circuit` to compile every circuit). The twin's QPM
   host then needs its own environment, built from
   `setup/qblox-twin-requirements.txt`: the standard requirements without the
   `iqm-*` packages, plus `qblox-scheduler==1.0.1`. That host cannot also run
   the IQM QPMs.

   ```bash
   uv venv --python 3.12 /path/to/qfw-qblox-venv
   source /path/to/qfw-qblox-venv/bin/activate
   uv pip install -r setup/build-requirements.txt \
     -r setup/qblox-twin-requirements.txt --override setup/overrides.txt
   source <prefix>/bin/qfw-activate --venv /path/to/qfw-qblox-venv
   ```

3. **Transmon `QuantumDevice` files** (YAML or JSON with
   `BasicTransmonElement`) also need `qblox_applications`, which is not on
   public PyPI and comes from Qblox. Option 1 avoids installing it on QFw
   hosts: run the script where it is already installed.
   `DeviceCompilationConfig` files need only qblox-scheduler, unless their
   operation factories live in `qblox_applications`.

`tests/test_qblox_twin_requirements.py` keeps `qblox-twin-requirements.txt`
in step with `requirements.txt`, and CI (`ci-resolve`) resolves it next to the
standard set.

## Tests

- `tests/test_qblox_twin.py`: the twin's engine and `QbloxTwinQPM` (needs
  numpy; the QuTiP cross-check runs when QuTiP is installed).
- `tests/test_qblox_timing.py`: Qblox timing, including a hardware compile on
  a dummy Cluster, and `bake_qblox_durations.py` (skips without
  qblox-scheduler).
- `tests/test_qblox_twin_requirements.py`: `setup/qblox-twin-requirements.txt`
  follows `setup/requirements.txt` without `iqm-*`, plus qblox-scheduler.
- `tests/mock/test_qblox_twin_qpm.py`: the service, with a stand-in twin,
  since the mock suite stubs numpy: query and directory resolution,
  calibration records, sync and async runs through a reserved QPM, failures,
  cancellation (queued and running) and shutdown.

## Modelling assumptions (change in qasm.py / calibration.py)

- Z-type gates are virtual (0 ns, no error); X/Y rotations are one pulse; U3 is two.
- `cx` = H(t) CZ H(t); `swap` = 3 `cx`. Errors are average gate errors, turned
  into depolarizing probabilities (p = 3r/2 for 1q, 5r/4 for 2q).
- Readout is classical confusion only; no mid-circuit measurement or reset.
- A CZ between uncoupled qubits runs with a default calibration and a warning
  (set `strict_connectivity` on `TwinBackend` to make it an error).
- Output bitstrings are Qiskit-ordered (highest classical bit leftmost),
  registers flattened without spaces. QFw's Qiskit client turns them into
  Qiskit memory and counts unchanged.

## Status

| Part | Status |
|---|---|
| QASM parser, lowering, noise model, sampling (numpy) | implemented, tested |
| QuTiP engine (`QFW_QBLOX_TWIN_ENGINE=qutip`) | implemented; matches numpy (Hellinger > 0.9999 on GHZ-3, QuTiP 5.3) |
| QFw service: registration, sync/async, cancel, telemetry | implemented, tested in the mock suite and end to end (Qiskit target → transpile → QPM → twin) with DEFw stubbed |
| Full-stack run (`qfw_ghz.sh … qblox-twin`) | not yet run against a live DEFw/PRTE deployment |
| `qblox_timing.py`: schedule, compile, per-op and per-element timing | implemented, tested with qblox-scheduler 1.0.1 (device layer and a dummy Cluster) |
| Q1Simulator check of the compiled Q1ASM | not done: q1simulator 1.4 needs Qt bindings even headless |
| `bake_qblox_durations.py` (Qblox durations baked into the calibration) | implemented, tested with qblox-scheduler 1.0.1, including a hardware compile |
| Twin host environment (`setup/qblox-twin-requirements.txt`) | resolves on Python 3.11, 3.12 and 3.13 (uv, and CI's `ci-resolve`); not yet installed end to end |
| Fork submodules | DEFw from `delgadoandrea21/DEFw` (pinned commit present); QHW packages from openQSE (pinned commits fetchable) |

## Remaining work

1. A first run on a live QFw (see "Running on a live QFw").
2. Readout duration: the twin does not time measurement yet, so the
   admission profile and the target carry no `measurement_ns`. Qblox timing
   already reports the `Measure` and `Reset` durations
   (`QbloxTwinQPM.qblox_durations`) for when it does.
3. Q1Simulator check of the compiled programs (real-time buffer underruns).
4. A hybrid (variational) example for the twin: the VQE example reads
   statevectors, which the twin does not return, so it needs a shot-based
   estimator (`QFwEstimatorV2`) and a layout fixed from the calibrated target.
