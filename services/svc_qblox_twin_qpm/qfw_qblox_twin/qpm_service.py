"""Plain-Python QPM facade over TwinBackend.

QFw registers it through ``services/svc_qblox_twin_qpm``: that service's QRC
creates, runs, cancels and deletes circuits here, and its QPM publishes
``capabilities()`` and the device calibration to QFw clients. Nothing in this
module imports QFw, so the twin also runs on its own (see the GHZ demo test in
``tests/test_qblox_twin.py``).
"""
from __future__ import annotations

import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Dict, List, Optional

from .calibration import DeviceCal
from .qasm import SUPPORTED_GATES
from .twin import TwinBackend


class QbloxTwinQPM:
    def __init__(self, device_path: Optional[str] = None, engine: str = "numpy",
                 seed: Optional[int] = None, max_workers: int = 2):
        device = DeviceCal.load(device_path) if device_path else DeviceCal.example_4q()
        self._backend = TwinBackend(device, engine=engine, seed=seed)
        self._lock = threading.Lock()          # TwinBackend shares one RNG
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="qblox-twin")
        self._circuits: Dict[str, dict] = {}
        self._futures: Dict[str, Future] = {}
        self._metadata: Dict[str, dict] = {}

    # ---- discovery ------------------------------------------------------
    @property
    def device(self) -> DeviceCal:
        return self._backend.device

    def capabilities(self) -> dict:
        d = self._backend.device
        return {
            "type": "simulator",
            "name": f"qblox-twin:{d.name}",
            "engine": self._backend.engine,
            "num_qubits": d.n_qubits,
            "coupling_map": [list(c.qubits) for c in d.couplers],
            "basis": ["rz", "rx", "ry", "cz"],
            # every OpenQASM 2 instruction parse_qasm accepts
            "instructions": list(SUPPORTED_GATES) + ["measure"],
            "input_format": "qasm2",
            "statevector": False,
        }

    # ---- circuit lifecycle ---------------------------------------------
    def create_circuit(self, qasm: str, shots: int = 1024) -> str:
        cid = uuid.uuid4().hex
        self._circuits[cid] = {"qasm": qasm, "shots": shots}
        return cid

    def delete_circuit(self, cid: str) -> None:
        self._circuits.pop(cid, None)
        self._futures.pop(cid, None)
        self._metadata.pop(cid, None)

    def read_queue(self) -> List[str]:
        return [c for c, f in self._futures.items() if not f.done()]

    # ---- execution ------------------------------------------------------
    def _execute(self, cid: str) -> dict:
        c = self._circuits[cid]
        with self._lock:
            res = self._backend.run_qasm(c["qasm"], c["shots"])
        self._metadata[cid] = dict(res.metadata)
        return res.counts

    def run(self, cid: str, sync: bool = True,
            on_done: Optional[Callable[[str, Future], None]] = None):
        """Run a created circuit.

        ``sync=True`` returns the counts. ``sync=False`` queues the run and
        returns ``cid``; ``get_result(cid)`` then waits for the counts.
        ``on_done(cid, future)``, if given, is called once the queued run
        finishes, fails or is cancelled. It may run on the pool's thread or
        at once on the caller of ``run()`` or ``cancel()``, so it should only
        hand the future off, never block.
        """
        if sync:
            return self._execute(cid)
        future = self._pool.submit(self._execute, cid)
        self._futures[cid] = future
        if on_done is not None:
            future.add_done_callback(lambda f: on_done(cid, f))
        return cid

    def get_result(self, cid: str, timeout: Optional[float] = None) -> dict:
        """Plain counts dict, which the README says QFw backends accept."""
        return self._futures[cid].result(timeout)

    def metadata(self, cid: str) -> dict:
        """Schedule metadata of the circuit's last run (device, engine,
        duration_ns, n_cz, n_pulses); empty until a run has finished."""
        return dict(self._metadata.get(cid, {}))

    def cancel(self, cid: str) -> bool:
        """Cancel a queued run. False if it has started, ended or was never
        queued: a run in progress always finishes."""
        future = self._futures.get(cid)
        return future is not None and future.cancel()

    def shutdown(self, wait: bool = False) -> None:
        """Take no more runs and cancel the queued ones."""
        self._pool.shutdown(wait=wait, cancel_futures=True)
