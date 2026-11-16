"""
The simulated chip: the physical parameters the QuTiP model runs on.

This is the "true device" behind the simulated Cluster, separate from the
Qblox device config that programs it. When the two agree, gates come out
right; when they don't (a pulse amplitude that doesn't match a qubit's drive
strength, a CZ amplitude off its resonance, a missing phase correction), the
same errors appear as on hardware.

JSON layout (units in the key names):

	{
	  "name": "matched to dev_cfg.json",
	  "readout_decay_fraction": 0.5,
	  "qubits": {
	    "q0": {"frequency_ghz": 6.0, "anharmonicity_ghz": -0.2,
	           "t1_us": 40, "t2_us": 30,
	           "drive_rad_per_ns_per_v": 0.25,
	           "drive_detuning_ghz": 0.0,
	           "flux_ghz_per_v": -1.7,
	           "readout_p01": 0.01, "readout_p10": 0.02}
	  },
	  "couplings": {"q0_q1": {"g_mhz": 8.8}}
	}

drive_rad_per_ns_per_v turns the drive envelope (volts at the simulated
output) into a Rabi rate. drive_detuning_ghz is the qubit's frequency minus
its drive clock. flux_ghz_per_v shifts the qubit's frequency per volt on its
flux line. Couplings are exchange couplings named parent_child, as in the
Qblox device config's edges. readout_decay_fraction is how much of each
qubit's integration window it spends decaying before it is read: 0 reads
instantly, 1 lets it decay through the whole window.
"""
import json
from dataclasses import asdict, dataclass, field

QUBIT_DEFAULTS = {
	"frequency_ghz": 6.0,
	"anharmonicity_ghz": -0.2,
	"t1_us": 40.0,
	"t2_us": 30.0,
	"drive_rad_per_ns_per_v": 0.0,
	"drive_detuning_ghz": 0.0,
	"flux_ghz_per_v": 0.0,
	"readout_p01": 0.01,
	"readout_p10": 0.02,
}


class SimDeviceError(ValueError):
	pass


@dataclass
class SimQubit:
	frequency_ghz: float = QUBIT_DEFAULTS["frequency_ghz"]
	anharmonicity_ghz: float = QUBIT_DEFAULTS["anharmonicity_ghz"]
	t1_us: float = QUBIT_DEFAULTS["t1_us"]
	t2_us: float = QUBIT_DEFAULTS["t2_us"]
	drive_rad_per_ns_per_v: float = QUBIT_DEFAULTS["drive_rad_per_ns_per_v"]
	drive_detuning_ghz: float = QUBIT_DEFAULTS["drive_detuning_ghz"]
	flux_ghz_per_v: float = QUBIT_DEFAULTS["flux_ghz_per_v"]
	readout_p01: float = QUBIT_DEFAULTS["readout_p01"]
	readout_p10: float = QUBIT_DEFAULTS["readout_p10"]

	def check(self, name):
		if self.t1_us <= 0 or self.t2_us <= 0:
			raise SimDeviceError(f"{name}: t1_us and t2_us must be positive")
		if self.t2_us > 2 * self.t1_us:
			raise SimDeviceError(f"{name}: t2_us cannot exceed 2 * t1_us")
		for key in ("readout_p01", "readout_p10"):
			value = getattr(self, key)
			if not 0 <= value < 0.5:
				raise SimDeviceError(f"{name}: {key} must be in [0, 0.5)")


@dataclass
class SimDevice:
	qubits: dict
	couplings: dict = field(default_factory=dict)
	name: str = "simulated chip"
	readout_decay_fraction: float = 0.5

	def check(self):
		if not self.qubits:
			raise SimDeviceError("the simulated chip has no qubits")
		for name, qubit in self.qubits.items():
			qubit.check(name)
		for edge, spec in self.couplings.items():
			parent, _, child = edge.partition("_")
			if parent not in self.qubits or child not in self.qubits:
				raise SimDeviceError(f"coupling {edge} names an unknown qubit")
			if "g_mhz" not in spec:
				raise SimDeviceError(f"coupling {edge} needs g_mhz")
		if not 0 <= self.readout_decay_fraction <= 1:
			raise SimDeviceError("readout_decay_fraction must be in [0, 1]")
		return self

	def to_dict(self):
		return {
			"name": self.name,
			"readout_decay_fraction": self.readout_decay_fraction,
			"qubits": {name: asdict(q) for name, q in self.qubits.items()},
			"couplings": {k: dict(v) for k, v in self.couplings.items()},
		}

	@classmethod
	def from_dict(cls, data):
		unknown_top = set(data) - {"name", "readout_decay_fraction", "qubits", "couplings"}
		if unknown_top:
			raise SimDeviceError(f"unknown keys: {sorted(unknown_top)}")
		qubits = {}
		for name, spec in (data.get("qubits") or {}).items():
			unknown = set(spec) - set(QUBIT_DEFAULTS)
			if unknown:
				raise SimDeviceError(f"{name}: unknown keys {sorted(unknown)}")
			qubits[name] = SimQubit(**{k: float(v) for k, v in spec.items()})
		return cls(qubits=qubits, couplings=dict(data.get("couplings") or {}),
			   name=data.get("name", "simulated chip"),
			   readout_decay_fraction=float(data.get("readout_decay_fraction", 0.5))).check()

	@classmethod
	def load(cls, path):
		with open(path) as handle:
			return cls.from_dict(json.load(handle))

	def save(self, path):
		with open(path, "w") as handle:
			json.dump(self.to_dict(), handle, indent=2)
			handle.write("\n")
