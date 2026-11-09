"""Optional QuTiP engine for the decoherence step.

Same physics as noise.decohere_numpy, but integrated with QuTiP's Lindblad
solver. This is the hook for richer models later (pulse-level drives,
crosstalk, ZZ). Cross-check against numpy with tests/test_twin.py when QuTiP
is installed.
"""
from __future__ import annotations

import math

import numpy as np


def decohere_qutip(rho, q, n, dt_ns, t1_us, t2_us):
    import qutip as qt

    if dt_ns <= 0:
        return rho
    t1 = t1_us * 1e3
    t2 = min(t2_us * 1e3, 2 * t1)
    gamma1 = 1 / t1 if math.isfinite(t1) else 0.0            # 1/ns
    gamma_phi = max(1 / t2 - 1 / (2 * t1), 0.0)              # 1/ns

    def on(op):  # embed single-qubit op on qubit q
        return qt.tensor(*[op if i == q else qt.qeye(2) for i in range(n)])

    c_ops = []
    if gamma1 > 0:
        c_ops.append(math.sqrt(gamma1) * on(qt.destroy(2)))
    if gamma_phi > 0:
        # L = sqrt(g/2) sigma_z dephases coherences at rate g
        c_ops.append(math.sqrt(gamma_phi / 2) * on(qt.sigmaz()))
    if not c_ops:
        return rho

    dim = 2 ** n
    dims = [[2] * n, [2] * n]
    rho0 = qt.Qobj(rho.reshape(dim, dim), dims=dims)
    H = 0 * qt.tensor(*[qt.qeye(2)] * n)
    res = qt.mesolve(H, rho0, [0.0, dt_ns], c_ops=c_ops)
    return np.asarray(res.states[-1].full()).reshape((2,) * (2 * n))
