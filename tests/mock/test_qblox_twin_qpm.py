# Guards the Qblox digital-twin QPM (svc_qblox_twin_qpm): that it presents
# the twin as a superconducting simulator with a device of its own, publishes
# the twin's calibration in the shapes the Qiskit client reads, and runs
# circuits in-process through UTIL_QRC's completion contract, sync and async,
# including cancellation and shutdown.
#
# The twin's numerics need a real numpy, which this suite stubs, so a fake
# with QbloxTwinQPM's interface and threading stands in for it here. The twin
# itself is tested in tests/test_qblox_twin.py.

import importlib.util
import pathlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import util.qpm.util_qpm as util_qpm
import util.qpm.util_qrc as util_qrc
from defw_exception import DEFwExecutionError
from tests.mock.fakes import FakeSchedulerContext
from tests.mock.test_fake_iqm_qpm import FakeAdmissionContext
from svc_qblox_twin_qpm.svc_qpm import QPM, calibration_set_id
from svc_qblox_twin_qpm.svc_qrc import QRC
from util.qpm.controller import _clear_target_controllers_for_tests
from util.qpm.util_circuit import Circuit, CircuitStates

SERVICE_DIR = (pathlib.Path(__file__).resolve().parents[2] /
	       "services" / "svc_qblox_twin_qpm")
DEVICE_ID = "qblox-twin-example-4q"
GHZ = ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\ncreg c[4];\n'
       'h q[0];\ncx q[0],q[1];\ncx q[1],q[2];\ncx q[2],q[3];\n'
       'measure q -> c;\n')
COUNTS = {"0000": 480, "1111": 470, "0001": 50}
METADATA = {"device": "example-4q", "engine": "numpy",
	    "duration_ns": 500.0, "n_cz": 3, "n_pulses": 7}


def _calibration_module():
	# The twin package imports numpy at load; its calibration model does
	# not, so it is loaded from its file on its own.
	name = "_qblox_twin_calibration"
	if name not in sys.modules:
		path = SERVICE_DIR / "qfw_qblox_twin" / "calibration.py"
		spec = importlib.util.spec_from_file_location(name, path)
		module = importlib.util.module_from_spec(spec)
		sys.modules[name] = module
		spec.loader.exec_module(module)
	return sys.modules[name]


class FakeTwin:
	"""QbloxTwinQPM's interface and threading, without the simulation.

	Queued runs go through a real thread pool, so cancel() and the done
	callback behave as they do on the twin. A run waits on gate when one is
	set, and raises error when one is set.
	"""

	def __init__(self, workers=1):
		self.device = _calibration_module().DeviceCal.example_4q()
		self.gate = None
		self.error = None
		self.started = []
		self.created = {}
		self.deleted = []
		self._pool = ThreadPoolExecutor(max_workers=workers)
		self._futures = {}
		self._metadata = {}

	def capabilities(self):
		return {
			"type": "simulator",
			"name": f"qblox-twin:{self.device.name}",
			"engine": "numpy",
			"num_qubits": self.device.n_qubits,
			"coupling_map": [list(c.qubits) for c in self.device.couplers],
			"basis": ["rz", "rx", "ry", "cz"],
			"instructions": ["rz", "rx", "ry", "h", "cx", "cz", "measure"],
			"input_format": "qasm2",
			"statevector": False,
		}

	def create_circuit(self, qasm, shots=1024):
		tid = f"t{len(self.created)}"
		self.created[tid] = (qasm, shots)
		return tid

	def delete_circuit(self, tid):
		self.deleted.append(tid)

	def _execute(self, tid):
		self.started.append(tid)
		if self.gate is not None:
			assert self.gate.wait(5), "test never opened the gate"
		if self.error is not None:
			raise self.error
		self._metadata[tid] = dict(METADATA)
		return dict(COUNTS)

	def run(self, tid, sync=True, on_done=None):
		if sync:
			return self._execute(tid)
		future = self._pool.submit(self._execute, tid)
		self._futures[tid] = future
		if on_done is not None:
			future.add_done_callback(lambda f: on_done(tid, f))
		return tid

	def metadata(self, tid):
		return dict(self._metadata.get(tid, {}))

	def cancel(self, tid):
		future = self._futures.get(tid)
		return future is not None and future.cancel()

	def shutdown(self, wait=False):
		self._pool.shutdown(wait=wait, cancel_futures=True)


class _Event:
	def __init__(self, evtype, payload):
		self.evtype = evtype
		self.payload = payload

	def get_evtype(self):
		return self.evtype

	def get_event(self):
		return self.payload


class _Sink:
	def __init__(self):
		self.events = []

	def put(self, event):
		self.events.append(event.get_event())
		return True


@pytest.fixture(autouse=True)
def _environment(monkeypatch):
	_clear_target_controllers_for_tests()
	for name in ("QFW_QPM_TARGET_ID", "QFW_QPU_DEVICE_ID",
		     "QFW_QBLOX_TWIN_DEVICE", "QFW_QBLOX_TWIN_ENGINE",
		     "QFW_QBLOX_TWIN_SEED", "QFW_SITE_CONFIG"):
		monkeypatch.delenv(name, raising=False)
	monkeypatch.setenv("QFW_QPM_ASSIGNED_HOSTS", "localhost:1")
	monkeypatch.setattr(util_qpm, "qpm_initialized", True)
	monkeypatch.setattr(util_qrc, "Event", _Event)


def _circuit(cid="c1", **overrides):
	info = {"qasm": GHZ, "num_qubits": 4, "num_shots": 1000,
		"qtask_id": f"q-{cid}", "qfw_backend": "qblox-twin"}
	info.update(overrides)
	freed = []
	circ = Circuit(cid, info, lambda c, result=None: freed.append(result))
	circ.set_ready()
	# As the QPM leaves it at dispatch. Circuit states only rise, and
	# EXEC_DONE sits below this one, so a success shows as a completion
	# time and a failure as FAIL.
	circ.set_resources_consumed()
	return circ, freed


def _close(value, expected):
	# pytest.approx looks at numpy, which this suite stubs.
	return abs(value - expected) < 1e-12


def _wait_for(predicate, timeout=2.0):
	deadline = time.monotonic() + timeout
	while time.monotonic() < deadline:
		value = predicate()
		if value:
			return value
		time.sleep(0.005)
	raise AssertionError("timed out waiting")


def _qpm(twin=None):
	return QPM(
		twin=twin or FakeTwin(),
		admission_context_factory=FakeAdmissionContext,
		scheduler_context_factory=FakeSchedulerContext,
	)


def _reserve(qpm, job="job-qblox-twin"):
	decision = qpm.reserve(request={
		"owner": {"user": "twin-user"},
		"job_id": job,
		"scope_id": "allocation-1",
		"target_device_id": DEVICE_ID,
		"walltime_ns": 1_000_000_000,
		"task_class": {
			"qubit_count": 4, "depth": 6, "one_q_gate_count": 7,
			"two_q_gate_count": 3, "measurement_count": 4,
			"shots": 1000,
		},
	})
	assert decision["status"] == "accepted", decision
	return decision["reservation_id"]


def _submission(**overrides):
	info = {"qasm": GHZ, "num_qubits": 4, "num_shots": 1000, "depth": 6,
		"one_q_gate_count": 7, "two_q_gate_count": 3,
		"measurement_count": 4}
	info.update(overrides)
	return info


def _wait_for_completion(qpm, cid, reservation_id, timeout=2.0):
	def ready():
		completion = qpm.read_cq(cid=cid, reservation_id=reservation_id)
		return completion if completion.get("completion_ready") else None
	return _wait_for(ready, timeout)


# ---- QRC -------------------------------------------------------------------

def test_sync_run_returns_counts_in_the_util_qrc_result_shape():
	twin = FakeTwin()
	qrc = QRC(start=False, twin=twin)
	circ, freed = _circuit()

	result = qrc.sync_run(circ)

	assert set(result) == {
		"cid", "result", "rc", "launch_time", "creation_time",
		"exec_time", "completion_time", "resources_consumed_time",
		"cq_enqueue_time", "cq_dequeue_time"}
	assert result["cid"] == "c1"
	assert result["result"] == COUNTS
	assert result["rc"] == 0
	assert twin.created == {"t0": (GHZ, 1000)}
	assert twin.deleted == ["t0"]
	assert circ.getState() == CircuitStates.RESOURCES_CONSUMED
	assert circ.completion_time >= circ.exec_time > 0
	# On the sync path the QPM, not the QRC, gives the resources back.
	assert freed == []
	assert qrc.get_task_metadata("c1") == {
		"cid": "c1", "qtask_id": "q-c1", "provider": "qblox-twin",
		"target_id": DEVICE_ID, **METADATA}
	assert qrc.get_task_timing("c1")["device_duration_ns"] == 500.0
	assert qrc.get_task_timing("unknown") == {}


def test_sync_run_failure_fails_the_circuit_and_frees_the_twin():
	twin = FakeTwin()
	twin.error = NotImplementedError("unsupported gate 'ccx'")
	qrc = QRC(start=False, twin=twin)
	circ, _freed = _circuit()

	with pytest.raises(DEFwExecutionError, match="unsupported gate 'ccx'"):
		qrc.sync_run(circ)

	assert circ.getState() == CircuitStates.FAIL
	assert twin.deleted == ["t0"]


def test_shots_come_from_num_shots_then_shots():
	twin = FakeTwin()
	qrc = QRC(start=False, twin=twin)
	circ, _freed = _circuit(num_shots=None)
	del circ.info["num_shots"]
	circ.info["shots"] = 64
	qrc.sync_run(circ)
	assert twin.created["t0"][1] == 64

	bad, _freed = _circuit("c2", num_shots=0)
	with pytest.raises(DEFwExecutionError, match="at least 1"):
		qrc.sync_run(bad)


def test_async_run_stores_the_completion_for_read_cq():
	twin = FakeTwin()
	qrc = QRC(twin=twin)
	circ, freed = _circuit()

	handle = qrc.async_run(circ)

	assert handle == "qblox-twin-q-c1"
	_wait_for(lambda: qrc.peak_cq("c1"))
	result = qrc.read_cq("c1")
	assert result["result"] == COUNTS
	assert result["rc"] == 0
	assert result["cq_dequeue_time"] >= result["cq_enqueue_time"]
	assert freed and freed[0]["cid"] == "c1"
	assert circ.getState() == CircuitStates.RESOURCES_CONSUMED
	assert circ.completion_time >= circ.exec_time > 0
	assert twin.deleted == ["t0"]
	assert qrc.get_task_metadata("c1")["n_cz"] == 3
	qrc.shutdown()


def test_async_run_pushes_the_completion_when_a_client_listens():
	qrc = QRC(twin=FakeTwin())
	sink = _Sink()
	qrc.register_event_notification({"class": sink, "evtype": "completion"})
	circ, _freed = _circuit()

	qrc.async_run(circ)

	_wait_for(lambda: sink.events)
	assert sink.events[0]["cid"] == "c1"
	assert sink.events[0]["result"] == COUNTS
	# Delivered, so not also kept for read_cq.
	assert qrc.read_cq("c1") is None
	qrc.shutdown()


def test_async_failure_reports_like_a_failed_runner():
	twin = FakeTwin()
	twin.error = ValueError("circuit needs 5 qubits, device has 4")
	qrc = QRC(twin=twin)
	circ, freed = _circuit()

	qrc.async_run(circ)

	result = _wait_for(lambda: qrc.read_cq("c1"))
	assert result["rc"] == 1
	assert result["result"].startswith("{result: ")
	assert "circuit needs 5 qubits" in result["result"]
	assert circ.getState() == CircuitStates.FAIL
	assert freed[0]["rc"] == 1
	assert twin.deleted == ["t0"]
	qrc.shutdown()


def test_cancel_stops_a_queued_run_but_not_a_running_one():
	twin = FakeTwin(workers=1)
	twin.gate = threading.Event()
	qrc = QRC(twin=twin)
	first, _freed = _circuit("c1")
	second, _freed = _circuit("c2")

	running = qrc.async_run(first)
	_wait_for(lambda: twin.started)
	queued = qrc.async_run(second)

	assert qrc.cancel(queued) == "cancelled"
	assert qrc.cancel(running) == "pending"
	assert qrc.cancel("qblox-twin-unknown") == "not-found"
	cancelled = _wait_for(lambda: qrc.read_cq("c2"))
	twin.gate.set()
	finished = _wait_for(lambda: qrc.read_cq("c1"))

	assert cancelled["reason"] == "provider-cancelled"
	assert cancelled["rc"] == 1
	assert second.getState() == CircuitStates.FAIL
	assert finished["rc"] == 0
	assert finished["result"] == COUNTS
	assert twin.started == ["t0"]
	assert sorted(twin.deleted) == ["t0", "t1"]
	qrc.shutdown()


def test_shutdown_reports_queued_runs_as_cancelled():
	twin = FakeTwin(workers=1)
	twin.gate = threading.Event()
	qrc = QRC(twin=twin)
	first, _freed = _circuit("c1")
	second, freed = _circuit("c2")
	qrc.async_run(first)
	_wait_for(lambda: twin.started)
	qrc.async_run(second)

	qrc.shutdown()

	cancelled = _wait_for(lambda: qrc.read_cq("c2"))
	assert cancelled["reason"] == "provider-cancelled"
	assert freed and freed[0]["cid"] == "c2"
	twin.gate.set()


def test_twin_from_environment_rejects_a_missing_device_file(
		monkeypatch, tmp_path):
	from defw_exception import DEFwError
	from svc_qblox_twin_qpm.svc_qrc import twin_from_environment

	monkeypatch.setenv("QFW_QBLOX_TWIN_DEVICE", str(tmp_path / "none.json"))
	with pytest.raises(DEFwError, match="is not a file"):
		twin_from_environment()
	monkeypatch.delenv("QFW_QBLOX_TWIN_DEVICE")
	monkeypatch.setenv("QFW_QBLOX_TWIN_SEED", "seven")
	with pytest.raises(DEFwError, match="must be an integer"):
		twin_from_environment()


# ---- QPM -------------------------------------------------------------------

def test_query_presents_a_superconducting_simulator_with_a_device():
	from api_qpm_common import QPMCapability, QPMType
	from qfw_qiskit.qpm_selection import qpm_selection_for_provider

	info = _qpm().query()
	properties = info["properties"]

	assert info["qpm_type"] == int(QPMType.QPM_TYPE_SIMULATOR)
	assert info["qpm_capabilities"] == int(
		QPMCapability.QPM_CAP_SUPERCONDUCTING)
	assert properties["provider"] == "qblox-twin"
	assert properties["device_id"] == DEVICE_ID
	assert properties["simulator"] is True
	assert properties["hardware"] is False
	assert properties["num_qubits"] == 4
	assert properties["coupling_map"] == [[0, 1], [1, 2], [2, 3]]
	assert properties["basis_gates"] == ["rz", "rx", "ry", "cz"]
	assert "measure" in properties["instructions"]
	assert properties["calibration_set_id"].startswith("example-4q:")
	assert info["service_id"] == f"qpm:qblox-twin:{DEVICE_ID}"
	assert DEVICE_ID in info["selector"]["resources"]
	assert "QBLOX-TWIN-4q" in info["selector"]["resources"]
	assert "qblox-twin" in info["selector"]["aliases"]
	# A QFwBackend(provider="qblox-twin") asks for exactly this QPM, and a
	# request for a statevector simulator does not match it.
	selection = qpm_selection_for_provider("qblox-twin")
	assert selection["qpm_type"] == info["qpm_type"]
	assert selection["qpm_capabilities"] == info["qpm_capabilities"]
	nwqsim = qpm_selection_for_provider("nwqsim")
	assert not (int(nwqsim["qpm_capabilities"]) & info["qpm_capabilities"])


def test_query_metadata_resolves_through_the_defw_directory():
	import defw_directory
	from api_qpm_common import QPMCapability, QPMType

	info = _qpm().query()
	directory = defw_directory.Directory()
	directory.register_service({
		"service_id": info["service_id"],
		"service_name": info["service_name"],
		"service_type": info["service_type"],
		"runtime_id": "qblox-twin-runtime",
		"peer_handle": "qblox-twin-peer",
		"endpoint": {
			"address": "qblox-twin.example",
			"listen_port": 8790,
			"pid": 123,
			"node_name": "qblox-twin",
			"hostname": "qblox-twin-head",
		},
		"api_bindings": info["api_bindings"],
		"selector": info["selector"],
		"properties": info["properties"],
		"capability": info["capability"],
		"qpm_type": info["qpm_type"],
		"qpm_capabilities": info["qpm_capabilities"],
	})

	def resolve(**filters):
		return {record["service_record"]["service_id"]
			for record in directory.resolve_services(
				service_type="qfw.qpm", **filters)}

	assert resolve(selector_resource=DEVICE_ID) == {info["service_id"]}
	assert resolve(selector_alias="qblox-twin") == {info["service_id"]}
	assert resolve(
		qpm_type=int(QPMType.QPM_TYPE_SIMULATOR),
		qpm_capabilities=int(QPMCapability.QPM_CAP_SUPERCONDUCTING),
	) == {info["service_id"]}
	assert resolve(
		qpm_capabilities=int(QPMCapability.QPM_CAP_STATEVECTOR)) == set()
	assert resolve(qpm_type=int(QPMType.QPM_TYPE_HARDWARE)) == set()


def test_the_deployment_names_the_device(monkeypatch):
	monkeypatch.setenv("QFW_QPU_DEVICE_ID", "lab-chip-7")

	info = _qpm().query()

	assert info["properties"]["device_id"] == "lab-chip-7"
	assert info["service_id"] == "qpm:qblox-twin:lab-chip-7"


def test_admission_profile_carries_the_twin_size_and_gate_times():
	qpm = _qpm()

	profile = qpm.controller.admission_context.registered_profiles[-1]

	assert profile["max_qubits"] == 4
	assert profile["max_shots"] == 1_000_000
	assert profile["one_q_gate_ns"] == 40
	assert profile["two_q_gate_ns"] == 120
	assert profile["external_device_id"] == DEVICE_ID


def test_device_records_publish_the_calibration():
	twin = FakeTwin()
	qpm = _qpm(twin)

	backend = qpm.get_backend_info()
	device = qpm.get_device_info()
	coupling = qpm.get_coupling_graph()
	snapshot = qpm.get_calibration_snapshot()

	assert backend["metadata_supported"] is True
	assert backend["target_id"] == DEVICE_ID
	assert device["num_qubits"] == 4
	assert coupling["couplings"] == [[0, 1], [1, 2], [2, 3]]
	assert coupling["directed"] is False
	operations = {op["name"]: op for op in coupling["operations"]}
	assert list(operations) == ["rz", "rx", "ry", "cz", "measure"]
	assert operations["cz"]["supported_loci"] == [[0, 1], [1, 2], [2, 3]]
	assert operations["rx"]["supported_loci"] == [[0], [1], [2], [3]]

	metrics = snapshot["extensions"]["qdmi.fomac.v1"]
	assert metrics["duration_unit"] == "ns"
	gates = {(m["operation"], tuple(m["locus"])): m
		 for m in metrics["gate_metrics"]}
	assert _close(gates[("cz", ("2", "3"))]["fidelity"], 0.98)
	assert gates[("cz", ("2", "3"))]["duration"] == 120.0
	assert _close(gates[("rx", ("3",))]["fidelity"], 0.999)
	assert gates[("rx", ("3",))]["duration"] == 40.0
	assert gates[("rz", ("0",))] == {
		"operation": "rz", "locus": ["0"], "fidelity": 1.0,
		"duration": 0.0}
	assert _close(gates[("measure", ("2",))]["fidelity"], 0.97)
	assert "duration" not in gates[("measure", ("2",))]
	assert metrics["qubit_metrics"][0] == {
		"qubit": "0", "t1": 45_000.0, "t2": 38_000.0}
	assert (snapshot["calibration_set_id"] ==
		backend["calibration_set_id"] ==
		coupling["calibration_set_id"])


def test_calibration_set_id_follows_the_numbers():
	device = FakeTwin().device
	before = calibration_set_id(device)

	device.couplers[0].cz_error = 0.004

	assert calibration_set_id(device) != before
	assert calibration_set_id(device).startswith("example-4q:")


def test_ideal_device_leaves_out_infinite_coherence():
	twin = FakeTwin()
	twin.device = twin.device.ideal_copy()
	snapshot = _qpm(twin).get_calibration_snapshot()

	metrics = snapshot["extensions"]["qdmi.fomac.v1"]
	assert metrics["qubit_metrics"][0] == {"qubit": "0"}


def test_reserved_async_run_completes_through_the_qpm():
	qpm = _qpm()
	reservation_id = _reserve(qpm)

	response = qpm.async_run(_submission(), reservation_id=reservation_id)
	_wait_for(lambda: qpm.task_status(
		qtask_id=response["qtask_id"],
		reservation_id=reservation_id).get("outcome") == "COMPLETED")
	metadata = qpm.get_task_metadata(
		reservation_id=reservation_id, task_id=response["qtask_id"])
	completion = _wait_for_completion(qpm, response["cid"], reservation_id)

	assert response["outcome"] == "ACCEPTED"
	assert response["provider_handle"] == f"qblox-twin-{response['qtask_id']}"
	# A UTIL_QRC result says nothing of its outcome when it succeeds, as
	# with NW-QSim; the controller marks only failures.
	assert "outcome" not in completion
	assert completion["result"] == COUNTS
	assert completion["rc"] == 0
	assert metadata["provider_metadata"]["n_cz"] == 3
	assert metadata["provider_metadata"]["target_id"] == DEVICE_ID
	assert qpm.free_hosts == {"localhost": 1}


def test_reserved_sync_run_returns_the_counts():
	qpm = _qpm()
	reservation_id = _reserve(qpm)

	response = qpm.sync_run(_submission(), reservation_id=reservation_id)

	assert response["outcome"] == "COMPLETED"
	assert response["result"]["result"] == COUNTS
	assert qpm.free_hosts == {"localhost": 1}


def test_failed_run_reaches_the_client_as_failed():
	twin = FakeTwin()
	twin.error = NotImplementedError("unsupported gate 'ccx'")
	qpm = _qpm(twin)
	reservation_id = _reserve(qpm)

	response = qpm.async_run(_submission(), reservation_id=reservation_id)
	completion = _wait_for_completion(qpm, response["cid"], reservation_id)

	assert completion["outcome"] == "FAILED"
	assert completion["rc"] == 1
	assert "unsupported gate 'ccx'" in completion["result"]
	assert qpm.free_hosts == {"localhost": 1}


def test_cancelling_a_running_circuit_waits_for_the_twin():
	# The controller calls the QRC's cancel() while it holds its lock. A
	# run the twin has started cannot stop, so the cancel stays pending
	# and the circuit completes; nothing deadlocks on the way.
	twin = FakeTwin()
	twin.gate = threading.Event()
	qpm = _qpm(twin)
	reservation_id = _reserve(qpm)
	response = qpm.async_run(_submission(), reservation_id=reservation_id)
	_wait_for(lambda: twin.started)

	status = qpm.cancel_task(
		cid=response["cid"], reservation_id=reservation_id)
	twin.gate.set()
	completion = _wait_for_completion(qpm, response["cid"], reservation_id)

	assert status["outcome"] == "CANCEL_PENDING"
	assert status["provider_cancel_status"] == "pending"
	assert completion["rc"] == 0
	assert completion["result"] == COUNTS
	assert qpm.free_hosts == {"localhost": 1}


def test_cancelling_a_queued_circuit_returns_its_slot(monkeypatch):
	# A QPM dispatches one circuit at a time unless the site raises its
	# limit. With two in flight, two slots and one twin worker, the second
	# waits in the twin's queue, where a cancel removes it. The twin reports
	# that on the cancelling thread, inside the controller's lock.
	monkeypatch.setenv("QFW_QPM_ASSIGNED_HOSTS", "node-a:1,node-b:1")
	twin = FakeTwin(workers=1)
	twin.gate = threading.Event()
	qpm = _qpm(twin)
	qpm.configure_dispatch_limits(limits={"max_inflight": 2})
	reservation_id = _reserve(qpm)
	running = qpm.async_run(_submission(), reservation_id=reservation_id)
	_wait_for(lambda: twin.started)
	queued = qpm.async_run(_submission(), reservation_id=reservation_id)
	assert "provider_handle" in queued, queued

	status = qpm.cancel_task(cid=queued["cid"], reservation_id=reservation_id)
	_wait_for(lambda: sum(qpm.free_hosts.values()) == 1)
	twin.gate.set()
	completion = _wait_for_completion(qpm, running["cid"], reservation_id)

	assert status["outcome"] == "CANCELLED"
	assert status["provider_cancel_status"] == "cancelled"
	assert completion["rc"] == 0
	assert twin.started == ["t0"]
	_wait_for(lambda: qpm.free_hosts == {"node-a": 1, "node-b": 1})
