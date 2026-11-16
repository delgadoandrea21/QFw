# svc_qblox_qpm

A QPM service for the Qblox control stack, built on `qblox-scheduler`'s
`HardwareAgent`. It is shaped exactly like `svc_iqm_qpm`:

| File | IQM analogue | Contents |
|---|---|---|
| `__init__.py` | `svc_iqm_qpm/__init__.py` | `svc_info`, `initialize`/`uninitialize` |
| `svc_qpm.py` | `svc_iqm_qpm/svc_qpm.py` | `QPM(UTIL_QPM)`, device profile, `query()` |
| `svc_qrc.py` | `svc_iqm_qpm/svc_qrc.py` | completion queue, sync/async dispatch |
| `util_qblox.py` | `util_iqm.py` | `QbloxServiceClient` — the control stack |
| `configs/` | — | `hw_cfg.json`, `dev_cfg.json` |
| `qblox_sim/` | — | the simulated Cluster (`qblox-sim`) |
| `derive_qblox_sim.py` | — | writes the simulated chip and calibrated config |

`svc_qrc.py` is provider-agnostic plumbing and is near-identical to IQM's.
Everything that knows what Qblox is lives in `util_qblox.py`.

## Three service IDs, one module

Which identity the process takes is decided by whether it is asked to
simulate, and otherwise by whether a cluster address is configured:

| Service | Selected by | QPM type | Lifecycle |
|---|---|---|---|
| `qblox-sim` | service name `qblox-sim`, or `QBLOX_SIMULATE=1` | simulator | allocation-local |
| `qblox-dummy` | no cluster address | simulator | allocation-local, like `fake-iqm` |
| `qblox-cluster` | `device-access.yaml` | hardware | `operation-mode: long-running` |

The dummy is `qblox-scheduler`'s built-in software cluster: pure software,
no device access, no route to an instrument, and placeholder readout. The
simulated Cluster runs the same compiled programs and returns physical
results; see below.

## Where the cluster address comes from

Service manifests have no `environment:` block, so the real cluster's
address reaches the service through the operator-owned device-access file.
QFw sets `QFW_QPU_DEVICE_ID` from the manifest's `device-id` and points
`QFW_DEVICE_ACCESS_CFG` at the file; the client then resolves it:

```yaml
qpus:
  qblox-cluster:
    provider: qblox
    provider-device-id: cluster0
    url: 192.0.2.10
```

There is no `credential-db`: a Qblox cluster is an address on a lab network,
not a web API with a token. That is why `credential-mode` is `no-secret`.

Resolution order, highest first:

1. a per-reservation credential carrying `cluster_ip`
2. `QBLOX_CLUSTER_IP` in the environment
3. `device-access.yaml`, when `device-id` and `QFW_DEVICE_ACCESS_CFG` are set
4. nothing — the dummy cluster

## Configuration

| Variable | Meaning |
|---|---|
| `QBLOX_CLUSTER_IP` | Cluster address. Unset selects the dummy cluster. |
| `QBLOX_CLUSTER_NAME` | Cluster name, default `cluster0`. |
| `QBLOX_HARDWARE_CONFIG` | Absolute path to `hw_cfg.json`. |
| `QBLOX_DEVICE_CONFIG` | Absolute path to `dev_cfg.json`. |
| `QBLOX_ENV_FILE` | Path to a `.env`; defaults to `.env` beside this file. |
| `QBLOX_PROFILE_STAGES` | `1` to record per-stage timings. |
| `QFW_QBLOX_MAX_QUBITS` | Override the count read from `dev_cfg.json`. |
| `QFW_QBLOX_MAX_SHOTS` | Shot ceiling, default 10000. |
| `QFW_QBLOX_TARGET_ID` | Override the service target id. |

Relative config paths resolve against `configs/`, never the working
directory: a QFw service starts in a generated run directory, on a different
node than the one that launched it.

A `.env` beside this file is read at import (see `.env.example`); short names
like `cluster_ip` map to their `QBLOX_*` equivalents, and `None` means unset.
The process environment wins over the file, so the manifest and
`device-access.yaml` always take precedence.

## Concurrency

`HardwareAgent` and the instrument coordinator's `stop()`/`prepare()`/
`start()` are global device operations — two circuits executing at once would
corrupt each other's acquisition. Enforced in three places:

- `max_ppn=1` on `UTIL_QPM`
- `concurrent_jobs: 1` in the device profile, so qhw-scheduler never
  dispatches two
- an explicit lock around the execute path in `QbloxServiceClient`

`async_run` means "the caller does not block", not "two circuits run at once".

## Circuits, schedule and results

`build_schedule` re-transpiles each circuit to `rx`, `ry` and `cz` and turns
it into a qblox-scheduler `Schedule`:

- **As soon as possible, per qubit.** Each operation waits only for the
  last operation on each of its own qubits, so gates on different qubits run
  in parallel. (qblox-scheduler's default, without a reference operation, is
  to start after the previously added operation, which runs a circuit one
  operation at a time.) A `barrier` makes the next operation on any of its
  qubits wait for all of them.
- **Readout together at the end.** All measurements start at once, after the
  last operation on every measured qubit. One final measurement per qubit;
  mid-circuit measurement is rejected.
- **Counts in Qiskit's order.** Keys have one character per classical bit,
  with classical bit 0 on the right, and each qubit's result goes to the
  classical bit the circuit measured it into. QFw's Qiskit client passes the
  keys through unchanged.

`tests/test_qblox_qpm.py` checks these, compiles against `configs/`, and runs
a circuit end to end on the dummy cluster (needs qiskit and qblox-scheduler
1.0.0b4, whose `BasicTransmonElement` the shipped `dev_cfg.json` uses).

## The simulated Cluster (`qblox-sim`)

`qblox-sim` compiles each circuit exactly as for a real Cluster, then runs
the Q1ASM in [q1simulator](https://github.com/sldesnoo-Delft/q1simulator)
and drives a QuTiP model of the chip with the result:

1. **Compile.** `build_schedule` and `HardwareAgent.compile()` with the
   service's hardware and device configs, as for hardware, one repetition.
2. **Run the Q1ASM** (`qblox_sim/q1run.py`). Every sequencer program runs
   unchanged in a simulated Cluster. Each sequencer renders I and Q at NCO
   frequency 0 with its phase registers applied, which is the drive in the
   frame of its clock, including every `set_ph_delta` (virtual Z). Ports map
   to qubits through the hardware config's connectivity; acquisitions map
   to channels through the compiled `acq_hardware_mapping`.
3. **Simulate the chip** (`qblox_sim/engine.py`). Three-level transmons with
   T1 and T2, driven by those envelopes; flux pulses shift a qubit's
   frequency; exchange couplings act while a flux pulse plays (idle ZZ is
   left out). QuTiP's `mesolve` integrates it, segment by segment.
4. **Read out.** Each measured qubit decays for `readout_decay_fraction` of
   its integration window (0.5 by default), level 2 reads as 1, and its
   assignment errors apply. Shots are sampled from the resulting
   probabilities and returned as +1/-1 thresholded values, the shape
   `HardwareAgent.run()` returns, so the counts code is shared.

### The simulated chip

The chip is what the configs program, not the configs themselves
(`qblox_sim/physics.py`): per qubit, frequency, anharmonicity, T1, T2, drive
strength (rad/ns per volt of envelope), drive detuning from its clock, flux
sensitivity and readout errors; per edge, a coupling in MHz.

With `QBLOX_SIM_DEVICE` unset, the service matches a chip to the device
config at start-up (about ten seconds, `qblox_sim/derive.py`) and calibrates
a copy of the config against it, as a lab would:

- each qubit's drive strength makes its `amp180` pulse a pi rotation;
- each CZ's flux amplitude brings `|11>` into resonance with `|20>`, and its
  length is one full exchange cycle, giving a conditional phase of pi;
- DRAG `beta` is scanned to remove the third level's Stark shift (it lands
  on 1/(2|anharmonicity|), 0.398 ns for -200 MHz);
- each CZ's `parent_phase_correction` and `child_phase_correction` are set
  to undo the single-qubit phases it leaves.

The shipped `dev_cfg.json` plays both of q0's CZs at the same flux amplitude,
which no chip can resonate with two partners. The calibrated copy plays
`q0_q2` at three times that amplitude, resonating through `|02>`, and the
start-up log says so. T1 (40 us), T2 (30 us), anharmonicity (-200 MHz) and
readout errors (1% and 2%) are defaults the configs don't set.

`derive_qblox_sim.py` writes the matched chip and the calibrated config to
files. Point the service at both to skip matching at start-up, or edit the
chip (a shorter T1, a qubit detuned from its clock) and keep the config to
see what that costs:

```bash
python derive_qblox_sim.py --out-device sim_device.json --out-config dev_cfg.sim.json
export QBLOX_SIM_DEVICE=$PWD/sim_device.json QBLOX_DEVICE_CONFIG=$PWD/dev_cfg.sim.json
```

| Variable | Meaning |
|---|---|
| `QBLOX_SIMULATE` | `1` runs the simulated Cluster whatever the service name; `0` never |
| `QBLOX_SIM_DEVICE` | simulated-chip JSON; unset matches one to the device config |
| `QBLOX_SIM_SEED` | integer seed for shot sampling |

With the shipped configs and the matched chip, GHZ-3 at 4000 shots puts
about 87% of shots on `000` and `111`. Most of the rest is readout: q0 and
q2 integrate for 5 us, so a qubit in `|1>` decays for 2.5 us before it is
read. One GHZ run takes about 1.5 s.

Needs `q1simulator` and `qutip` (in `setup/requirements.txt`); q1simulator
needs Python 3.12 or later and a Qt binding, which qblox-scheduler 1.0.0b4
brings. `tests/qiskit/test_qblox_sim.py` covers the Q1ASM rendering, the
matched chip and calibration, gate and readout probabilities, and a run
through the service.

**Limits.** Exchange couplings act only during flux pulses, so idle ZZ and
idle swaps are absent. A flux pulse shifts frequency linearly with
amplitude. Readout is a projection with decay and assignment errors, not a
resonator simulation. All measurements must be at the end, as
`build_schedule` already requires. Matching spreads at most two CZs per
flux line. Every simulated qubit has three levels, so cost grows as 3^n per
active qubit: three qubits take about a second, five much longer.

## Known gaps

- **Device-profile timings are placeholders.** `QBLOX_PROFILE_STAGES=1`
  records real per-stage timings through `get_task_timing()`; sweep qubit
  count and depth and derive the coefficients from that.
- **The shipped `dev_cfg.json` declares 3 qubits** (q0-q2, CZ edges q0_q1
  and q0_q2, though its `name` is `device_2q`), with `acq_rotation` and
  `acq_threshold` both 0 (uncalibrated). Circuits wider than 3 qubits are
  rejected by admission, which is correct.
- **No cancellation.** Qblox has no mid-schedule abort, so `cancel()` can
  only stop a queued task from starting.
- **Dummy readout is degenerate.** `set_dummy_binned_acquisition_data` is not
  wired up, so the dummy is a plumbing test, not a believable GHZ
  distribution.
