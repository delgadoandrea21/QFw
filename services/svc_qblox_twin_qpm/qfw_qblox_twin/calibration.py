"""Device calibration data model (the 'twin' parameters).

Units: times in us/ns as named; errors are *average gate errors* (RB-style),
probabilities are in [0, 1].
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class QubitCal:
    t1_us: float = 30.0
    t2_us: float = 20.0
    readout_p01: float = 0.01      # P(read 1 | prepared 0)
    readout_p10: float = 0.02      # P(read 0 | prepared 1)
    sq_error: float = 5e-4         # average error per single-qubit pulse
    sq_duration_ns: float = 40.0   # duration of one single-qubit pulse


@dataclass
class CouplerCal:
    qubits: Tuple[int, int] = (0, 1)
    cz_error: float = 5e-3         # average error of one CZ
    cz_duration_ns: float = 100.0


@dataclass
class DeviceCal:
    name: str = "twin"
    qubits: Dict[int, QubitCal] = field(default_factory=dict)
    couplers: List[CouplerCal] = field(default_factory=list)

    @property
    def n_qubits(self) -> int:
        return len(self.qubits)

    def coupler(self, a: int, b: int) -> Optional[CouplerCal]:
        key = frozenset((a, b))
        for c in self.couplers:
            if frozenset(c.qubits) == key:
                return c
        return None

    # ---------------------------------------------------------------- io
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "qubits": {str(k): asdict(v) for k, v in self.qubits.items()},
            "couplers": [
                {**asdict(c), "qubits": list(c.qubits)} for c in self.couplers
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DeviceCal":
        return cls(
            name=d.get("name", "twin"),
            qubits={int(k): QubitCal(**v) for k, v in d["qubits"].items()},
            couplers=[
                CouplerCal(**{**c, "qubits": tuple(c["qubits"])})
                for c in d.get("couplers", [])
            ],
        )

    @classmethod
    def load(cls, path: str | Path) -> "DeviceCal":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    # ------------------------------------------------------- constructors
    def ideal_copy(self) -> "DeviceCal":
        """Same topology and timings, but no decoherence or errors."""
        qubits = {
            k: QubitCal(t1_us=math.inf, t2_us=math.inf, readout_p01=0.0,
                        readout_p10=0.0, sq_error=0.0,
                        sq_duration_ns=v.sq_duration_ns)
            for k, v in self.qubits.items()
        }
        couplers = [CouplerCal(c.qubits, 0.0, c.cz_duration_ns)
                    for c in self.couplers]
        return DeviceCal(f"{self.name}-ideal", qubits, couplers)

    @classmethod
    def example_4q(cls) -> "DeviceCal":
        """4-qubit line 0-1-2-3 with deliberately uneven quality.

        Pair (0,1) is good, pair (2,3) is poor. Illustrative numbers only,
        not measured data.
        """
        qubits = {
            0: QubitCal(t1_us=45, t2_us=38, readout_p01=0.008, readout_p10=0.015, sq_error=3e-4),
            1: QubitCal(t1_us=40, t2_us=32, readout_p01=0.010, readout_p10=0.018, sq_error=4e-4),
            2: QubitCal(t1_us=22, t2_us=14, readout_p01=0.020, readout_p10=0.040, sq_error=8e-4),
            3: QubitCal(t1_us=15, t2_us=9,  readout_p01=0.030, readout_p10=0.060, sq_error=1e-3),
        }
        couplers = [
            CouplerCal((0, 1), cz_error=3e-3, cz_duration_ns=80),
            CouplerCal((1, 2), cz_error=8e-3, cz_duration_ns=100),
            CouplerCal((2, 3), cz_error=2e-2, cz_duration_ns=120),
        ]
        return cls("example-4q", qubits, couplers)
