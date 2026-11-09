import hashlib
import json
import logging
import math

from .svc_qrc import QRC, QBLOX_TWIN_BACKEND
from util.qpm.util_qpm import UTIL_QPM
from util.qpm.util_circuit import set_max_qubits_pp

MAX_QBLOX_TWIN_SHOTS = 1_000_000
# A device runs one circuit at a time, and the twin runs one at a time under
# its lock anyway. One slot per host keeps the queue in QFw's scheduler, where
# admission and cancellation see it, not in the twin's thread pool.
QBLOX_TWIN_MAX_PPN = 1
# The native operations the twin times and charges errors for (see
# qfw_qblox_twin/qasm.py): rz is a virtual frame change, rx and ry are one
# pulse each, cz is the two-qubit gate. Clients that build a device target
# compile to these.
QBLOX_TWIN_ONE_Q_NATIVE = ("rz", "rx", "ry")
QBLOX_TWIN_TWO_Q_NATIVE = ("cz",)
# The calibration snapshot uses the layout of the qdmi.fomac.v1 extension,
# which QFw's Qiskit client reads into the target's errors and durations.
CALIBRATION_EXTENSION = "qdmi.fomac.v1"


class QPM(UTIL_QPM):
	def __init__(self, start=True, twin=None, admission_context_factory=None,
		     scheduler_context_factory=None):
		qrc = QRC(start=start, twin=twin)
		# UTIL_QPM drops self.qrc at shutdown; the device stays describable.
		self.twin = qrc.twin
		self.device_id = qrc.device_id
		super().__init__(
			qrc,
			max_ppn=QBLOX_TWIN_MAX_PPN,
			start=start,
			target_id=self.device_id,
			admission_context_factory=admission_context_factory,
			scheduler_context_factory=scheduler_context_factory)
		device = self.twin.device
		set_max_qubits_pp(device.n_qubits)
		self.configure_device_profile(**qblox_twin_profile(device))

	def query(self):
		from . import SERVICE_NAME, SERVICE_DESC
		from api_qpm_common import QPMType, QPMCapability
		capabilities = self.twin.capabilities()
		info = self.query_helper(
			QPMType.QPM_TYPE_SIMULATOR,
			QPMCapability.QPM_CAP_SUPERCONDUCTING,
			SERVICE_NAME, SERVICE_DESC,
			properties={
				"provider": QBLOX_TWIN_BACKEND,
				"device_id": self.device_id,
				"target_id": self.device_id,
				"num_qubits": capabilities["num_qubits"],
				"max_shots": MAX_QBLOX_TWIN_SHOTS,
				"instructions": capabilities["instructions"],
				"basis_gates": capabilities["basis"],
				"coupling_map": capabilities["coupling_map"],
				"engine": capabilities["engine"],
				# nominal, qblox-durations or qblox-circuit: where the
				# twin's gate times come from (see svc_qrc.py)
				"timing": capabilities.get("timing", "nominal"),
				"calibration_set_id": calibration_set_id(self.twin.device),
			})
		logging.debug(f"Qblox twin {SERVICE_DESC}: {info}")
		return info

	def prepare_circuit(self, info):
		info['qfw_backend'] = QBLOX_TWIN_BACKEND
		return info

	# ---- device telemetry (api_qpm_telemetry) ---------------------------
	# NW-QSim has no device and answers metadata_supported False. The twin
	# has one, so it publishes its calibration in the shapes
	# backends/qfw_qiskit/qfw_target.py reads, and a QFwBackend compiles to
	# the twin's coupling map and native gates with its errors and durations.
	def get_backend_info(self, lib=None, token=None, reservation_id=None):
		info = self._device_record()
		info["calibration_set_id"] = calibration_set_id(self.twin.device)
		return info

	def get_device_info(self, lib=None, token=None):
		info = self._device_record()
		info.update({
			"max_shots": MAX_QBLOX_TWIN_SHOTS,
			"engine": self.twin.capabilities()["engine"],
			"calibration_name": self.twin.device.name,
		})
		return info

	def get_dynamic_backend_info(self, calibration_set_id=None, lib=None,
				     token=None):
		info = self.get_device_info(lib=lib, token=token)
		info["calibration_set_id"] = _current_calibration(
			self.twin.device, calibration_set_id)
		return info

	def get_calibration_snapshot(self, calibration_set_id=None, lib=None,
				     token=None):
		device = self.twin.device
		record = self._device_record()
		current = _current_calibration(device, calibration_set_id)
		record.update({
			"calibration_set_id": current,
			"calibration": {"calibration_set_id": current},
			"extensions": {
				CALIBRATION_EXTENSION: qblox_twin_calibration_metrics(device),
			},
		})
		return record

	def get_coupling_graph(self, calibration_set_id=None, lib=None,
			       token=None):
		device = self.twin.device
		record = self._device_record()
		record.update({
			"calibration_set_id": _current_calibration(
				device, calibration_set_id),
			"directed": False,
			"couplings": [list(c.qubits) for c in device.couplers],
			"operations": qblox_twin_operations(device),
		})
		return record

	def _device_record(self):
		return {
			"backend": QBLOX_TWIN_BACKEND,
			"provider": QBLOX_TWIN_BACKEND,
			"target_id": self.device_id,
			"device_id": self.device_id,
			"num_qubits": self.twin.device.n_qubits,
			"metadata_supported": True,
		}


def qblox_twin_profile(device):
	"""Admission device profile: the twin's size and its gate times."""
	one_q_ns = [q.sq_duration_ns for q in device.qubits.values()]
	two_q_ns = [c.cz_duration_ns for c in device.couplers]
	profile = {
		"max_qubits": device.n_qubits,
		"max_shots": MAX_QBLOX_TWIN_SHOTS,
	}
	if one_q_ns:
		profile["one_q_gate_ns"] = max(1, math.ceil(max(one_q_ns)))
	if two_q_ns:
		profile["two_q_gate_ns"] = max(1, math.ceil(max(two_q_ns)))
	return profile


def calibration_set_id(device):
	"""A stable id for the loaded calibration: changes when its numbers do."""
	text = json.dumps(device.to_dict(), sort_keys=True, default=str)
	digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
	return f"{device.name}:{digest}"


def _current_calibration(device, requested):
	# The twin holds one calibration. A request for another gets the
	# current one, with its own id, so the caller can tell.
	current = calibration_set_id(device)
	if requested not in (None, current):
		logging.debug(
			f"qblox twin has calibration {current}, not {requested}")
	return current


def qblox_twin_operations(device):
	qubits = [[q] for q in sorted(device.qubits)]
	operations = [
		{"name": name, "arity": 1, "supported_loci": qubits}
		for name in QBLOX_TWIN_ONE_Q_NATIVE]
	pairs = [list(c.qubits) for c in device.couplers]
	if pairs:
		operations.extend(
			{"name": name, "arity": 2, "supported_loci": pairs}
			for name in QBLOX_TWIN_TWO_Q_NATIVE)
	operations.append(
		{"name": "measure", "arity": 1, "supported_loci": qubits})
	return operations


def qblox_twin_calibration_metrics(device):
	"""Per-gate fidelity and duration, and per-qubit T1/T2, in nanoseconds.

	Fidelity is 1 - the average gate error the twin charges; readout is
	1 - the mean of its two confusion probabilities. Non-finite values (an
	ideal device) are left out rather than sent as infinities.
	"""
	gates, qubits = [], []
	for q, cal in sorted(device.qubits.items()):
		locus = [str(q)]
		gates.append({"operation": "rz", "locus": locus,
			      "fidelity": 1.0, "duration": 0.0})
		for name in ("rx", "ry"):
			gates.append(_metric(name, locus, 1.0 - cal.sq_error,
					     cal.sq_duration_ns))
		gates.append(_metric(
			"measure", locus,
			1.0 - (cal.readout_p01 + cal.readout_p10) / 2.0, None))
		coherence = {"qubit": str(q)}
		for key, value in (("t1", cal.t1_us), ("t2", cal.t2_us)):
			if math.isfinite(value):
				coherence[key] = value * 1e3
		qubits.append(coherence)
	for coupler in device.couplers:
		gates.append(_metric(
			"cz", [str(q) for q in coupler.qubits],
			1.0 - coupler.cz_error, coupler.cz_duration_ns))
	return {
		"duration_unit": "ns",
		"gate_metrics": gates,
		"qubit_metrics": qubits,
	}


def _metric(operation, locus, fidelity, duration):
	entry = {"operation": operation, "locus": locus}
	if math.isfinite(fidelity):
		entry["fidelity"] = fidelity
	if duration is not None and math.isfinite(duration):
		entry["duration"] = float(duration)
	return entry
