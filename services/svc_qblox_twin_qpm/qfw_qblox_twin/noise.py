"""Density-matrix primitives (numpy) and noise channels.

State layout: rho is a tensor of shape (2,)*(2n); axes 0..n-1 are the ket
indices of qubits 0..n-1, axes n..2n-1 the bra indices. Qubit 0 is axis 0.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

I2 = np.eye(2, dtype=complex)
PX = np.array([[0, 1], [1, 0]], dtype=complex)
PY = np.array([[0, -1j], [1j, 0]], dtype=complex)
PZ = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULIS = [I2, PX, PY, PZ]


def zero_state(n: int) -> np.ndarray:
    rho = np.zeros((2,) * (2 * n), dtype=complex)
    rho[(0,) * (2 * n)] = 1.0
    return rho


def apply_1q(rho, K, q, n):
    """rho -> K rho K^dagger on qubit q (K need not be unitary)."""
    out = np.tensordot(K, rho, axes=([1], [q]))
    out = np.moveaxis(out, 0, q)
    out = np.tensordot(out, K.conj(), axes=([n + q], [1]))
    return np.moveaxis(out, -1, n + q)


def apply_2q(rho, K, a, b, n):
    """rho -> K rho K^dagger for a 4x4 K acting on (a, b); a is the MSB."""
    K4 = K.reshape(2, 2, 2, 2)
    out = np.tensordot(K4, rho, axes=([2, 3], [a, b]))
    out = np.moveaxis(out, [0, 1], [a, b])
    out = np.tensordot(out, K4.conj(), axes=([n + a, n + b], [2, 3]))
    return np.moveaxis(out, [-2, -1], [n + a, n + b])


CZ = np.diag([1, 1, 1, -1]).astype(complex)


# --------------------------------------------------------------- channels
def depolarize_1q(rho, q, n, avg_error):
    """Depolarizing channel with the given average gate error (d=2: p = 3r/2)."""
    if avg_error <= 0:
        return rho
    p = min(1.5 * avg_error, 0.75)
    out = (1 - p) * rho
    for P in (PX, PY, PZ):
        out = out + (p / 3) * apply_1q(rho, P, q, n)
    return out


def depolarize_2q(rho, a, b, n, avg_error):
    """Two-qubit depolarizing channel (d=4: p = 5r/4)."""
    if avg_error <= 0:
        return rho
    p = min(1.25 * avg_error, 15 / 16)
    out = (1 - p) * rho
    for i, j in itertools.product(range(4), repeat=2):
        if i == 0 and j == 0:
            continue
        out = out + (p / 15) * apply_2q(rho, np.kron(_PAULIS[i], _PAULIS[j]), a, b, n)
    return out


def decohere_numpy(rho, q, n, dt_ns, t1_us, t2_us):
    """T1 amplitude damping + pure dephasing for dt_ns on qubit q (exact)."""
    if dt_ns <= 0:
        return rho
    t1 = t1_us * 1e3
    t2 = min(t2_us * 1e3, 2 * t1)                 # physical limit T2 <= 2 T1
    gamma = 1 - math.exp(-dt_ns / t1) if math.isfinite(t1) else 0.0
    rate_phi = max(1 / t2 - 1 / (2 * t1), 0.0)    # 1/T_phi (0 if t's are inf)
    lam = 1 - math.exp(-2 * dt_ns * rate_phi)
    if gamma > 0:
        k0 = np.array([[1, 0], [0, math.sqrt(1 - gamma)]], dtype=complex)
        k1 = np.array([[0, math.sqrt(gamma)], [0, 0]], dtype=complex)
        rho = apply_1q(rho, k0, q, n) + apply_1q(rho, k1, q, n)
    if lam > 0:
        k0 = np.array([[1, 0], [0, math.sqrt(1 - lam)]], dtype=complex)
        k1 = np.array([[0, 0], [0, math.sqrt(lam)]], dtype=complex)
        rho = apply_1q(rho, k0, q, n) + apply_1q(rho, k1, q, n)
    return rho
