"""Minimal OpenQASM 2.0 parser + lowering to a native {1q pulse, CZ} op list.

Supported: qreg/creg, barrier, terminal measure, and the gates listed in
``_ONE_Q`` plus cx, cz, swap. Custom gate definitions, mid-circuit
measurement, reset and classical control raise NotImplementedError.
"""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------- data types
@dataclass
class NativeOp:
    kind: str                       # "sq" or "cz"
    qubits: Tuple[int, ...]
    unitary: Optional[np.ndarray] = None   # 2x2 for "sq"
    pulses: int = 0                 # physical pulses (0 = virtual/frame change)
    label: str = ""


@dataclass
class Circuit:
    n_qubits: int
    n_cbits: int
    ops: List[NativeOp]
    measures: Dict[int, int]        # cbit -> qubit


# ------------------------------------------------------------------ gates
def _rz(phi):
    return np.diag([np.exp(-0.5j * phi), np.exp(0.5j * phi)])


def _rx(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def _ry(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _p(lam):
    return np.diag([1, np.exp(1j * lam)])


def _u3(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])


_SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
_H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)

# name -> (n_params, unitary(*params), pulses).  Pulse counts are MODELLING
# ASSUMPTIONS for a transmon with virtual-Z: Z-type gates are free, any
# X/Y rotation is one pulse, a general U3 is two.
_ONE_Q = {
    "id":   (0, lambda: np.eye(2, dtype=complex), 0),
    "x":    (0, lambda: _rx(math.pi), 1),
    "y":    (0, lambda: _ry(math.pi) , 1),
    "z":    (0, lambda: np.diag([1, -1]).astype(complex), 0),
    "h":    (0, lambda: _H, 1),
    "s":    (0, lambda: np.diag([1, 1j]), 0),
    "sdg":  (0, lambda: np.diag([1, -1j]), 0),
    "t":    (0, lambda: np.diag([1, np.exp(1j * math.pi / 4)]), 0),
    "tdg":  (0, lambda: np.diag([1, np.exp(-1j * math.pi / 4)]), 0),
    "sx":   (0, lambda: _SX, 1),
    "sxdg": (0, lambda: _SX.conj().T, 1),
    "rx":   (1, _rx, 1),
    "ry":   (1, _ry, 1),
    "rz":   (1, _rz, 0),
    "p":    (1, _p, 0),
    "u1":   (1, _p, 0),
    "u2":   (2, lambda phi, lam: _u3(math.pi / 2, phi, lam), 1),
    "u3":   (3, _u3, 2),
    "u":    (3, _u3, 2),
}
_TWO_Q = {"cx", "cz", "swap"}

# Every gate name parse_qasm accepts (measure and barrier are statements).
SUPPORTED_GATES = tuple(_ONE_Q) + tuple(sorted(_TWO_Q))


def _sq(name, params, q) -> NativeOp:
    n, fn, pulses = _ONE_Q[name]
    if len(params) != n:
        raise ValueError(f"gate {name} expects {n} params, got {len(params)}")
    return NativeOp("sq", (q,), np.asarray(fn(*params), dtype=complex), pulses, name)


def _cz(a, b) -> NativeOp:
    return NativeOp("cz", (a, b), None, 0, "cz")


def _lower(name: str, params: List[float], qs: List[int]) -> List[NativeOp]:
    if name in _ONE_Q:
        return [_sq(name, params, qs[0])]
    if name == "cz":
        return [_cz(*qs)]
    if name == "cx":
        c, t = qs
        return [_sq("h", [], t), _cz(c, t), _sq("h", [], t)]
    if name == "swap":
        a, b = qs
        return (_lower("cx", [], [a, b]) + _lower("cx", [], [b, a])
                + _lower("cx", [], [a, b]))
    raise NotImplementedError(f"unsupported gate '{name}'")


# ----------------------------------------------------------------- parsing
_SAFE_FUNCS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan}


def _eval_expr(expr: str) -> float:
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name) and n.id == "pi":
            return math.pi
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.Div):
                return a / b
            if isinstance(n.op, ast.Pow):
                return a ** b
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id in _SAFE_FUNCS and len(n.args) == 1):
            return _SAFE_FUNCS[n.func.id](ev(n.args[0]))
        raise ValueError(f"unsupported parameter expression: {expr!r}")
    return ev(ast.parse(expr.strip(), mode="eval"))


def _split_top(s: str) -> List[str]:
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        if ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return [x.strip() for x in out]


def parse_qasm(text: str) -> Circuit:
    text = re.sub(r"//[^\n]*", "", text)
    if re.search(r"(^|\s)(gate|opaque|if)\s", text):
        raise NotImplementedError("gate definitions / classical control not supported")
    qregs: Dict[str, Tuple[int, int]] = {}
    cregs: Dict[str, Tuple[int, int]] = {}
    nq = nc = 0
    ops: List[NativeOp] = []
    measures: Dict[int, int] = {}
    measured_qubits = set()

    def resolve(tok: str, regs):
        m = re.fullmatch(r"(\w+)\[(\d+)\]", tok)
        if m:
            off, size = regs[m.group(1)]
            i = int(m.group(2))
            if i >= size:
                raise ValueError(f"index out of range: {tok}")
            return [off + i]
        off, size = regs[tok]
        return list(range(off, off + size))

    for stmt in (s.strip() for s in text.split(";")):
        if not stmt or stmt.startswith(("OPENQASM", "include")):
            continue
        m = re.fullmatch(r"(qreg|creg)\s+(\w+)\[(\d+)\]", stmt)
        if m:
            kind, name, size = m.group(1), m.group(2), int(m.group(3))
            if kind == "qreg":
                qregs[name] = (nq, size)
                nq += size
            else:
                cregs[name] = (nc, size)
                nc += size
            continue
        if stmt.startswith("barrier"):
            continue
        if stmt.startswith("reset"):
            raise NotImplementedError("reset not supported")
        m = re.fullmatch(r"measure\s+(\S+)\s*->\s*(\S+)", stmt)
        if m:
            qs, cs = resolve(m.group(1), qregs), resolve(m.group(2), cregs)
            if len(qs) != len(cs):
                raise ValueError(f"size mismatch: {stmt}")
            for q, c in zip(qs, cs):
                if c in measures or q in measured_qubits:
                    raise NotImplementedError("each qubit/cbit may be measured once")
                measures[c] = q
                measured_qubits.add(q)
            continue
        m = re.fullmatch(r"(\w+)\s*(?:\((.*)\))?\s+(.+)", stmt)
        if not m:
            raise ValueError(f"cannot parse statement: {stmt!r}")
        name = m.group(1).lower()
        params = [_eval_expr(p) for p in _split_top(m.group(2))] if m.group(2) else []
        operands = [resolve(t, qregs) for t in _split_top(m.group(3))]
        width = max(len(o) for o in operands)
        for i in range(width):
            qs = [o[i] if len(o) > 1 else o[0] for o in operands]
            if any(q in measured_qubits for q in qs):
                raise NotImplementedError("gates after measurement not supported")
            ops.extend(_lower(name, params, qs))
    return Circuit(nq, nc, ops, measures)
