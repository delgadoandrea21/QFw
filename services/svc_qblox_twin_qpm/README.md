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

The calibration set id the QPM publishes is the calibration's name plus a
hash of its numbers, so it changes whenever the JSON does.

## Running

```
cd examples
./qfw_ghz.sh qiskit 4 qblox-twin 1
```

From Python, with a reservation as in `examples/tests/test_qiskit_ghz.py`:

```python
from qfw_qiskit import QFwBackend
backend = QFwBackend(provider="qblox-twin")
job = backend.run(backend.compile(circuit), shots=4096, reservation_id=...)
```

`backend.run(circuit)` without `compile` also works: the twin lowers
`h`, `cx`, `swap`, `u3` and the other gates in `capabilities()["instructions"]`
itself. Compiling first is what applies the calibration-aware layout.

## Tests

- `tests/test_qblox_twin.py`: the twin's engine and `QbloxTwinQPM` (needs
  numpy; the QuTiP cross-check runs when QuTiP is installed).
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
| `qblox_timing.py` (Qblox Scheduler / Q1Simulator durations) | stub only |

## Remaining work

1. Implement `qfw_qblox_twin/qblox_timing.py` against your qblox-scheduler version.
2. Readout duration: the twin does not time measurement, so the admission
   profile and the target carry no `measurement_ns`.
3. Hybrid loop on top of the calibration-aware layout.
