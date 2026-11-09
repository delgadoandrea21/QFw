"""TwinBackend: OpenQASM 2.0 in, noisy counts out.

Pipeline: parse -> lower to {1q pulse, CZ} -> ASAP schedule per qubit with
T1/T2 decoherence during gates and idle gaps -> per-pulse depolarizing error
-> readout confusion -> multinomial sampling.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from . import noise
from .calibration import CouplerCal, DeviceCal
from .qasm import Circuit, parse_qasm


@dataclass
class TwinResult:
    counts: Dict[str, int]
    shots: int
    metadata: dict = field(default_factory=dict)


def hellinger_fidelity(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    sp, sq = sum(p.values()) or 1.0, sum(q.values()) or 1.0
    return sum(math.sqrt(p.get(k, 0) / sp * q.get(k, 0) / sq) for k in keys) ** 2


class TwinBackend:
    def __init__(self, device: DeviceCal, engine: str = "numpy",
                 seed: Optional[int] = None, strict_connectivity: bool = False):
        if engine not in ("numpy", "qutip"):
            raise ValueError("engine must be 'numpy' or 'qutip'")
        self.device = device
        self.engine = engine
        self.strict = strict_connectivity
        self.rng = np.random.default_rng(seed)
        self._warned_pairs = set()
        if engine == "qutip":
            from .qutip_engine import decohere_qutip
            self._decohere = decohere_qutip
        else:
            self._decohere = noise.decohere_numpy

    # ------------------------------------------------------------- public
    def run_qasm(self, qasm: str, shots: int = 1024) -> TwinResult:
        probs, meta = self.exact_probabilities(qasm)
        keys = list(probs)
        p = np.array([probs[k] for k in keys])
        draws = self.rng.multinomial(shots, p / p.sum())
        counts = {k: int(c) for k, c in zip(keys, draws) if c > 0}
        return TwinResult(counts, shots, meta)

    def exact_probabilities(self, qasm: str):
        """Shot-noise-free outcome distribution {bitstring: prob} + metadata."""
        circ = parse_qasm(qasm)
        return self._simulate(circ)

    # ----------------------------------------------------------- internals
    def _cal_coupler(self, a, b) -> CouplerCal:
        c = self.device.coupler(a, b)
        if c is not None:
            return c
        if self.strict:
            raise ValueError(f"no coupler between qubits {a} and {b}; "
                             "transpile to the device coupling map first")
        key = frozenset((a, b))
        if key not in self._warned_pairs:
            warnings.warn(f"no coupler ({a},{b}) in device; using default CZ cal")
            self._warned_pairs.add(key)
        return CouplerCal((a, b))

    def _simulate(self, circ: Circuit):
        n = circ.n_qubits
        if n > self.device.n_qubits:
            raise ValueError(f"circuit needs {n} qubits, device has {self.device.n_qubits}")
        cal = self.device.qubits
        rho = noise.zero_state(n)
        clock = [0.0] * n
        n_cz = n_pulses = 0

        def advance(qubits, start, dur):
            nonlocal rho
            for q in qubits:
                idle = start - clock[q]
                if idle > 0:
                    rho = self._decohere(rho, q, n, idle, cal[q].t1_us, cal[q].t2_us)
                clock[q] = start

        for op in circ.ops:
            if op.kind == "sq":
                q = op.qubits[0]
                dur = cal[q].sq_duration_ns * op.pulses
                advance([q], clock[q], dur)
                half = dur / 2
                rho = self._decohere(rho, q, n, half, cal[q].t1_us, cal[q].t2_us)
                rho = noise.apply_1q(rho, op.unitary, q, n)
                rho = self._decohere(rho, q, n, half, cal[q].t1_us, cal[q].t2_us)
                for _ in range(op.pulses):
                    rho = noise.depolarize_1q(rho, q, n, cal[q].sq_error)
                clock[q] += dur
                n_pulses += op.pulses
            else:  # cz
                a, b = op.qubits
                cc = self._cal_coupler(a, b)
                start = max(clock[a], clock[b])
                advance([a, b], start, cc.cz_duration_ns)
                half = cc.cz_duration_ns / 2
                for q in (a, b):
                    rho = self._decohere(rho, q, n, half, cal[q].t1_us, cal[q].t2_us)
                rho = noise.apply_2q(rho, noise.CZ, a, b, n)
                for q in (a, b):
                    rho = self._decohere(rho, q, n, half, cal[q].t1_us, cal[q].t2_us)
                rho = noise.depolarize_2q(rho, a, b, n, cc.cz_error)
                clock[a] = clock[b] = start + cc.cz_duration_ns
                n_cz += 1

        # align all qubits at readout time
        end = max(clock) if clock else 0.0
        advance(range(n), end, 0.0)

        # outcome probabilities, readout confusion, marginalise to measured
        dim = 2 ** n
        p = np.real(np.diag(rho.reshape(dim, dim))).reshape((2,) * n)
        p = np.clip(p, 0, None)
        p /= p.sum()
        for q in range(n):
            c = cal[q]
            M = np.array([[1 - c.readout_p01, c.readout_p10],
                          [c.readout_p01, 1 - c.readout_p10]])
            p = np.moveaxis(np.tensordot(M, p, axes=([1], [q])), 0, q)

        cbits = sorted(circ.measures)
        if not cbits:
            raise ValueError("circuit has no measurements")
        qs = [circ.measures[c] for c in cbits]
        others = tuple(q for q in range(n) if q not in qs)
        pm = p.sum(axis=others) if others else p
        order = sorted(qs)
        pm = pm.transpose([order.index(q) for q in qs])

        probs: Dict[str, float] = {}
        for idx, val in np.ndenumerate(pm):
            if val < 1e-12:      # prune numerical dust
                continue
            bits = ["0"] * circ.n_cbits
            for c, b in zip(cbits, idx):
                bits[circ.n_cbits - 1 - c] = str(b)   # highest cbit leftmost
            probs["".join(bits)] = probs.get("".join(bits), 0.0) + float(val)
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
        meta = {"device": self.device.name, "engine": self.engine,
                "duration_ns": end, "n_cz": n_cz, "n_pulses": n_pulses}
        return probs, meta
