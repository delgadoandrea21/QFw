"""svc_qblox_qpm's schedule and counts (services/svc_qblox_qpm/util_qblox.py).

Here rather than in tests/mock because it needs the real Qiskit.

Needs qiskit, numpy, xarray and qblox-scheduler; skips without them. The
compile and dummy-cluster tests load the service's own dev_cfg.json, whose
BasicTransmonElement ships with qblox-scheduler 1.0.0b4 (QFw's pin) but not
with later releases, so they skip there.
"""
import importlib.util
import json
import os

import pytest

pytest.importorskip("numpy")
pytest.importorskip("xarray")
pytest.importorskip("qiskit")
pytest.importorskip("qblox_scheduler")

import xarray as xr  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SERVICE = os.path.join(ROOT, "services", "svc_qblox_qpm")
CONFIGS = os.path.join(SERVICE, "configs")


def _load_util_qblox():
	# Loaded on its own: the package's __init__ starts QPM services.
	spec = importlib.util.spec_from_file_location(
		"svc_qblox_qpm_util_qblox", os.path.join(SERVICE, "util_qblox.py"))
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


uq = _load_util_qblox()


def _compile(schedule):
	from qblox_scheduler import HardwareAgent
	from qblox_scheduler.backends.graph_compilation import SerialCompiler

	with open(os.path.join(CONFIGS, "hw_cfg.json")) as handle:
		hardware = json.load(handle)
	with open(os.path.join(CONFIGS, "dev_cfg.json")) as handle:
		device = json.load(handle)
	try:
		agent = HardwareAgent(hardware, device)
	except Exception as exc:
		pytest.skip(f"dev_cfg.json does not load with this qblox-scheduler: {exc}")
	compiled = SerialCompiler().compile(
		schedule=schedule,
		config=agent.quantum_device.generate_compilation_config())
	table = compiled.timing_table.data
	return table[table["abs_time"] > 0]  # after the reset


def _starts_ns(table, prefix):
	rows = table[table["operation"].str.startswith(prefix)]
	return [round(t * 1e9, 3) for t in rows["abs_time"]]


def _dataset(values):
	return xr.Dataset({q: ("repetition", v) for q, v in values.items()})


# --- counts -----------------------------------------------------------------

def test_counts_put_classical_bit_zero_on_the_right():
	# X on q0 only: Qiskit reads that as "001".
	values = {0: [1, 1], 1: [-1, -1], 2: [-1, -1]}
	assert uq.build_counts(values, {0: 0, 1: 1, 2: 2}, 3, 2) == {"001": 2}
	# Measured into classical bit 2 instead, it is the leftmost character.
	assert uq.build_counts(values, {0: 2, 1: 1, 2: 0}, 3, 2) == {"100": 2}


def test_counts_cover_only_measured_qubits():
	values = uq.extract_channel_values(_dataset({1: [1, -1, 1]}), [1])
	assert uq.build_counts(values, {1: 0}, 1, 3) == {"1": 2, "0": 1}
	# A classical bit no measurement writes stays 0.
	assert uq.build_counts(values, {1: 1}, 2, 3) == {"10": 2, "00": 1}


def test_build_schedule_maps_qubits_to_classical_bits():
	circuit = QuantumCircuit(3, 2)
	circuit.x(1)
	circuit.measure(1, 0)
	circuit.measure(2, 1)
	_, measured = uq.build_schedule(circuit, 4)
	assert measured == {1: 0, 2: 1}


def test_mid_circuit_and_repeated_measurements_are_rejected():
	repeated = QuantumCircuit(1, 2)
	repeated.measure(0, 0)
	repeated.measure(0, 1)
	with pytest.raises(Exception, match="measured twice"):
		uq.build_schedule(repeated, 4)

	mid = QuantumCircuit(2, 2)
	mid.measure(0, 0)
	mid.cz(0, 1)
	mid.measure(1, 1)
	with pytest.raises(Exception, match="after its measurement"):
		uq.build_schedule(mid, 4)

	with pytest.raises(Exception, match="measures no qubits"):
		uq.build_schedule(QuantumCircuit(1), 4)


# --- schedule timing (compiled with the service's own configs) --------------

def test_independent_qubits_run_together_and_readouts_start_together():
	circuit = QuantumCircuit(3, 3)
	circuit.rx(0.5, 1)
	circuit.rx(0.5, 2)
	circuit.cz(0, 1)
	circuit.measure([0, 1, 2], [0, 1, 2])
	table = _compile(uq.build_schedule(circuit, 4)[0])

	rx = _starts_ns(table, "Rxy")
	assert len(rx) == 2 and rx[0] == rx[1]            # q1 and q2 in parallel
	cz = _starts_ns(table, "SquarePulse(amplitude=0.1")  # the flux pulse
	assert cz == [rx[0] + 40]                           # after q1's 40 ns pulse
	reads = _starts_ns(table, "ThresholdedAcquisition")
	assert len(reads) == 3 and len(set(reads)) == 1     # all three together
	assert reads[0] >= cz[0] + 40                       # after the CZ


def test_barrier_holds_its_qubits_together():
	circuit = QuantumCircuit(3, 3)
	circuit.rx(0.5, 1)
	circuit.barrier(1, 2)
	circuit.rx(0.5, 2)
	circuit.measure([0, 1, 2], [0, 1, 2])
	table = _compile(uq.build_schedule(circuit, 4)[0])

	first, second = sorted(_starts_ns(table, "Rxy"))
	assert second == first + 40


def test_dummy_cluster_runs_a_circuit_end_to_end(monkeypatch, tmp_path):
	from util.circuit_payload import encode_qiskit_circuit, qiskit_circuit_formats

	monkeypatch.delenv("QBLOX_CLUSTER_IP", raising=False)
	monkeypatch.chdir(tmp_path)  # the dummy cluster writes tmp.q1asm and tmp.hex
	circuit = QuantumCircuit(3, 2)
	circuit.h(0)
	circuit.cx(0, 1)
	circuit.measure([0, 1], [0, 1])
	info = dict(encode_qiskit_circuit(circuit, qiskit_circuit_formats()))
	info["num_shots"] = 8

	class Circ:
		provider_credential = None

		def __init__(self, info):
			self.info = info

		def get_cid(self):
			return "c1"

	client = uq.QbloxServiceClient()
	try:
		client.agent()
	except Exception as exc:
		pytest.skip(f"dev_cfg.json does not load with this qblox-scheduler: {exc}")
	result = client.run_circuit(Circ(info))

	# The dummy cluster returns no real readout; check the shape only.
	assert sum(result["counts"].values()) == 8
	assert all(len(key) == 2 for key in result["counts"])
	assert sorted(result["qblox"]["per_shot_channel_values"]) == [0, 1]
	assert result["qblox"]["mode"] == "dummy"
