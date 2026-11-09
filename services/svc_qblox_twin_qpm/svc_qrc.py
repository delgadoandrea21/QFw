import logging
import os
import queue
import threading
import time
import traceback
from collections import OrderedDict

from defw_exception import DEFwError, DEFwExecutionError
from util import instrumentation
from util.circuit_payload import openqasm2_text
from util.qpm.util_qrc import UTIL_QRC

QBLOX_TWIN_BACKEND = "qblox-twin"

# The twin this service wraps is configured from the environment, which the
# service manager passes to the QPM process.
#   QFW_QBLOX_TWIN_DEVICE  calibration JSON (see device.example.json); unset
#                          runs DeviceCal.example_4q()
#   QFW_QBLOX_TWIN_ENGINE  numpy (default) or qutip
#   QFW_QBLOX_TWIN_SEED    integer seed for shot sampling; unset is unseeded
# Timing from the Qblox compile flow (needs qblox-scheduler; see
# qfw_qblox_twin/qblox_timing.py):
#   QFW_QBLOX_TWIN_QBLOX_DEVICE    QuantumDevice YAML/JSON, or a
#                                  DeviceCompilationConfig JSON; unset keeps
#                                  the calibration file's gate durations
#   QFW_QBLOX_TWIN_QBLOX_HARDWARE  Qblox hardware config JSON (optional)
#   QFW_QBLOX_TWIN_TIMING          durations (default): gate durations from
#                                  Qblox once, at start; circuit: also
#                                  compile and time every circuit
#   QFW_QBLOX_TWIN_QBLOX_ELEMENTS  device element per twin qubit, comma
#                                  separated; unset is q0,q1,...
DEVICE_ENV = "QFW_QBLOX_TWIN_DEVICE"
ENGINE_ENV = "QFW_QBLOX_TWIN_ENGINE"
SEED_ENV = "QFW_QBLOX_TWIN_SEED"
QBLOX_DEVICE_ENV = "QFW_QBLOX_TWIN_QBLOX_DEVICE"
QBLOX_HARDWARE_ENV = "QFW_QBLOX_TWIN_QBLOX_HARDWARE"
TIMING_ENV = "QFW_QBLOX_TWIN_TIMING"
QBLOX_ELEMENTS_ENV = "QFW_QBLOX_TWIN_QBLOX_ELEMENTS"
TIMING_MODES = ("durations", "circuit")
# The same names the QPM controller reads for its target id. A device-id in
# the services file arrives as QFW_QPU_DEVICE_ID.
TARGET_ID_ENV = "QFW_QPM_TARGET_ID"
DEVICE_ID_ENV = "QFW_QPU_DEVICE_ID"

DEFAULT_SHOTS = 1024
# Per-circuit metadata kept for get_task_timing and get_task_metadata.
MAX_REMEMBERED_TASKS = 1024


def twin_from_environment():
	"""The QbloxTwinQPM this service runs, configured as described above."""
	device = os.environ.get(DEVICE_ENV, "").strip() or None
	if device is not None and not os.path.isfile(device):
		raise DEFwError(f"{DEVICE_ENV}={device} is not a file")
	engine = os.environ.get(ENGINE_ENV, "").strip().lower() or "numpy"
	seed = os.environ.get(SEED_ENV, "").strip()
	try:
		seed = int(seed) if seed else None
	except ValueError:
		raise DEFwError(f"{SEED_ENV} must be an integer: {seed!r}") from None
	qblox = _qblox_options_from_environment()
	# Imported here, not at the top: the twin needs a real numpy, and the
	# service module should load without one.
	from .qfw_qblox_twin.qpm_service import QbloxTwinQPM

	logging.debug(
		f"qblox twin: device={device or 'example-4q'} engine={engine} "
		f"seed={seed} qblox={qblox}")
	return QbloxTwinQPM(device, engine=engine, seed=seed, **qblox)


def _qblox_options_from_environment():
	def path(env_name):
		value = os.environ.get(env_name, "").strip() or None
		if value is not None and not os.path.isfile(value):
			raise DEFwError(f"{env_name}={value} is not a file")
		return value

	qblox_device = path(QBLOX_DEVICE_ENV)
	qblox_hardware = path(QBLOX_HARDWARE_ENV)
	timing = os.environ.get(TIMING_ENV, "").strip().lower()
	elements = os.environ.get(QBLOX_ELEMENTS_ENV, "").strip()
	if timing and timing not in TIMING_MODES:
		raise DEFwError(
			f"{TIMING_ENV} must be one of {', '.join(TIMING_MODES)}: {timing!r}")
	if qblox_device is None:
		for env_name, value in ((QBLOX_HARDWARE_ENV, qblox_hardware),
					(TIMING_ENV, timing), (QBLOX_ELEMENTS_ENV, elements)):
			if value:
				raise DEFwError(f"{env_name} needs {QBLOX_DEVICE_ENV}")
		return {}
	options = {
		"qblox_device": qblox_device,
		"qblox_hardware": qblox_hardware,
		"qblox_per_circuit": timing == "circuit",
	}
	if elements:
		options["element_names"] = {
			index: name.strip()
			for index, name in enumerate(elements.split(","))}
	return options


def qblox_twin_device_id(twin):
	"""The id this QPM serves the twin under: the target or device id the
	deployment gave it, else one named for the calibration it loaded."""
	for env_name in (TARGET_ID_ENV, DEVICE_ID_ENV):
		value = os.environ.get(env_name, "").strip()
		if value:
			return value
	return f"{QBLOX_TWIN_BACKEND}-{twin.device.name}"


def _shots(info):
	shots = info.get("num_shots", info.get("shots", DEFAULT_SHOTS))
	try:
		shots = int(shots)
	except (TypeError, ValueError):
		raise DEFwExecutionError(f"num_shots must be an integer: {shots!r}") from None
	if shots < 1:
		raise DEFwExecutionError(f"num_shots must be at least 1: {shots}")
	return shots


class QRC(UTIL_QRC):
	"""Runs circuits on the Qblox digital twin, inside the QPM process.

	NW-QSim's QRC writes each circuit to a file and launches
	circuit_runner.nwqsim under MPI, and UTIL_QRC's runner threads poll the
	launcher for it. The twin is a Python object, so there is nothing to
	launch: this QRC hands circuits to QbloxTwinQPM and keeps only UTIL_QRC's
	completion queue (read_cq, peak_cq and the event push), the contract the
	QPM reads results through. Results have the shape UTIL_QRC gives them.

	An async run's completion is queued and finished on one thread of its
	own. The QPM controller calls cancel() while it holds its lock, and the
	twin may report a cancelled run on the caller's thread, so no QPM work
	happens where the twin calls back.
	"""

	def __init__(self, start=True, twin=None):
		# No launcher and no runner threads: nothing here is launched.
		super().__init__(start=False)
		self.start = start
		self.twin = twin if twin is not None else twin_from_environment()
		self.device_id = qblox_twin_device_id(self.twin)
		self._tasks_lock = threading.Lock()
		self._tasks = {}
		self._remembered = OrderedDict()
		self._done = queue.Queue()
		self._finisher = None

	# ---- runs ----------------------------------------------------------
	def sync_run(self, circ):
		info = circ.info
		qasm = openqasm2_text(info)
		shots = _shots(info)
		execution = self._begin(circ)
		circ.set_launching()
		tid = self.twin.create_circuit(qasm, shots)
		try:
			circ.set_running()
			counts = self.twin.run(tid, sync=True)
			metadata = self.twin.metadata(tid)
		except Exception as e:
			circ.set_fail()
			failure = DEFwExecutionError(
				f"Circuit failed on the qblox twin: {type(e).__name__}: {e}")
			instrumentation.finish_backend_execution(execution, circ, error=failure)
			raise failure from e
		finally:
			self.twin.delete_circuit(tid)
		circ.set_exec_done()
		instrumentation.finish_backend_execution(execution, circ)
		self._remember(circ, metadata)
		return self._result(circ, counts, 0)

	def async_run(self, circ):
		info = circ.info
		qasm = openqasm2_text(info)
		shots = _shots(info)
		handle = f"{QBLOX_TWIN_BACKEND}-{info.get('qtask_id') or circ.get_cid()}"
		execution = self._begin(circ)
		circ.set_launching()
		tid = self.twin.create_circuit(qasm, shots)
		with self._tasks_lock:
			self._tasks[handle] = {
				"circ": circ, "tid": tid, "telemetry": execution}
		self._start_finisher()
		circ.set_running()
		try:
			self.twin.run(
				tid, sync=False,
				on_done=lambda _tid, future: self._done.put((handle, future)))
		except Exception as e:
			with self._tasks_lock:
				self._tasks.pop(handle, None)
			self.twin.delete_circuit(tid)
			instrumentation.finish_backend_execution(execution, circ, error=e)
			raise
		return handle

	def cancel(self, provider_handle):
		with self._tasks_lock:
			task = self._tasks.get(provider_handle)
		if task is None:
			return "not-found"
		if self.twin.cancel(task["tid"]):
			return "cancelled"
		# Already running: it finishes and reports as usual.
		return "pending"

	def shutdown(self):
		logging.critical("shutdown called")
		self.shutdown_workers = True
		# Queued runs are cancelled, and reported, before the finisher stops.
		self.twin.shutdown(wait=False)
		self._done.put(None)

	# ---- provider details the QPM asks for ------------------------------
	def get_task_timing(self, cid=None):
		with self._tasks_lock:
			record = self._remembered.get(cid)
		return dict(record["timing"]) if record else {}

	def get_task_metadata(self, cid=None):
		with self._tasks_lock:
			record = self._remembered.get(cid)
		return dict(record["metadata"]) if record else {}

	# ---- completion ----------------------------------------------------
	def _start_finisher(self):
		with self._tasks_lock:
			if self._finisher is not None:
				return
			self._finisher = threading.Thread(
				target=self._finish_loop, name="qblox-twin-finisher")
			self._finisher.daemon = True
			self._finisher.start()

	def _finish_loop(self):
		while True:
			item = self._done.get()
			if item is None:
				return
			handle, future = item
			try:
				self._finish(handle, future)
			except Exception:
				# The error belongs to this circuit; the finisher carries
				# on with the others (as UTIL_QRC's runners do).
				logging.critical(
					f"finishing qblox twin task {handle} failed:\n"
					f"{traceback.format_exc()}")

	def _finish(self, handle, future):
		with self._tasks_lock:
			task = self._tasks.pop(handle, None)
		if task is None:
			return
		circ, tid = task["circ"], task["tid"]
		error, reason = None, None
		try:
			if future.cancelled():
				reason = "provider-cancelled"
				error = DEFwExecutionError("cancelled before the twin ran it")
			elif future.exception() is not None:
				e = future.exception()
				error = DEFwExecutionError(
					f"Circuit failed on the qblox twin: {type(e).__name__}: {e}")
			if error is None:
				output, rc = future.result(), 0
				circ.set_exec_done()
				self._remember(circ, self.twin.metadata(tid))
			else:
				# The text UTIL_QRC reports for a runner that failed.
				output, rc = "{result: " + f"{error}" + "}", 1
				circ.set_fail()
		finally:
			self.twin.delete_circuit(tid)
			instrumentation.finish_backend_execution(
				task["telemetry"], circ, error=error)

		r = self._result(circ, output, rc)
		if reason:
			r["reason"] = reason
		logging.defw_service(f"QPM task result: {r}")
		try:
			circ.free_resources(circ, result=r)
		finally:
			# The client gets its result even if the QPM's own
			# bookkeeping for it failed.
			self._push_or_store_result(r)

	# ---- helpers -------------------------------------------------------
	def _begin(self, circ):
		return instrumentation.begin_backend_execution(
			circ, instrumentation.API_PATH_SIMULATOR,
			device=self._device_name(circ),
			backend_kind=self._backend_kind(circ))

	def _device_name(self, circ):
		return self.device_id

	def _result(self, circ, output, rc):
		return {
			'cid': circ.get_cid(),
			'result': output,
			'rc': rc,
			'launch_time': circ.launch_time,
			'creation_time': circ.creation_time,
			'exec_time': circ.exec_time,
			'completion_time': circ.completion_time,
			'resources_consumed_time': circ.resources_consumed_time,
			'cq_enqueue_time': time.time(),
			'cq_dequeue_time': -1,
		}

	def _remember(self, circ, metadata):
		cid = circ.get_cid()
		record = {
			"timing": {
				"cid": cid,
				"qtask_id": circ.info.get("qtask_id"),
				"creation_time": circ.creation_time,
				"launch_time": circ.launch_time,
				"resources_consumed_time": circ.resources_consumed_time,
				"exec_time": circ.exec_time,
				"completion_time": circ.completion_time,
				# The twin's ASAP schedule, not wall-clock time.
				"device_duration_ns": metadata.get("duration_ns"),
			},
			"metadata": {
				"cid": cid,
				"qtask_id": circ.info.get("qtask_id"),
				"provider": QBLOX_TWIN_BACKEND,
				"target_id": self.device_id,
				**metadata,
			},
		}
		with self._tasks_lock:
			self._remembered[cid] = record
			while len(self._remembered) > MAX_REMEMBERED_TASKS:
				self._remembered.popitem(last=False)

	def test(self):
		return "****Testing the Qblox twin QRC****"
