"""Optional hook: take gate durations from the real Qblox compile flow.

STATUS: STUB. Not implemented or tested; I could not verify the
qblox-scheduler / Q1Simulator API for your installed version.

Today the twin takes durations from DeviceCal (sq_duration_ns, cz_duration_ns).
The intended upgrade, once you have a QuantumDevice + hardware config:

  1. Build a qblox_scheduler Schedule from the lowered native ops
     (qasm.parse_qasm(...).ops): Rxy for 1q pulses, CZ, Measure. Note the
     package was renamed from quantify_scheduler; import whichever you have.
  2. Compile with the compiler/hardware config for a (dummy) Cluster.
  3. Read per-operation start/stop times from the compiled schedule's
     timing table and replace the nominal durations used in twin._simulate.
  4. Optionally load the compiled Q1ASM into q1simulator's Cluster to check
     real-time-buffer underruns (it raises on FIFO errors).

Keep the interface narrow so the rest of the twin never imports Qblox code.
"""
from __future__ import annotations

from typing import Dict, Tuple

from .qasm import Circuit


def durations_from_qblox_scheduler(circuit: Circuit, quantum_device, hardware_config
                                   ) -> Dict[Tuple[int, str], float]:
    raise NotImplementedError(
        "Implement against your qblox-scheduler version; see module docstring.")
