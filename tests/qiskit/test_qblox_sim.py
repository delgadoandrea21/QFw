"""svc_qblox_qpm's simulated Cluster (services/svc_qblox_qpm/qblox_sim).

Compiles with the service's own configs, runs the Q1ASM in q1simulator and
simulates the chip in QuTiP. Needs qiskit, xarray, q1simulator, qutip and
qblox-scheduler 1.0.0b4 (the configs use its BasicTransmonElement); skips
without them. Matching a chip to the configs takes about ten seconds, once.
"""
import copy
import importlib.util
import json
import os
import sys

import pytest

for module in ("numpy", "xarray", "qiskit", "qblox_scheduler", "q1simulator", "qutip"):
	try:
		importlib.import_module(module)
	except Exception as exc:  # q1simulator raises a non-ImportError without Qt bindings
		pytest.skip(f"{module} unavailable: {exc}", allow_module_level=True)

import numpy as np  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SERVICE = os.path.join(ROOT, "services", "svc_qblox_qpm")
CONFIGS = os.path.join(SERVICE, "configs")
if SERVICE not in sys.path:
	sys.path.insert(0, SERVICE)

from qblox_sim import derive, engine, q1run  # noqa: E402
from qblox_sim.cluster import SimulatedCluster  # noqa: E402
from qblox_sim.physics import SimDevice, SimDeviceError  # noqa: E402


def _load(name):
	with open(os.path.join(CONFIGS, name)) as handle:
		return json.load(handle)


def _util_qblox():
	spec = importlib.util.spec_from_file_location(
		"svc_qblox_qpm_util_qblox_sim", os.path.join(SERVICE, "util_qblox.py"))
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


uq = _util_qblox()


@pytest.fixture(scope="module")
def configs():
	hw, dev = _load("hw_cfg.json"), _load("dev_cfg.json")
	try:
		derive.make_agent(hw, dev)
	except Exception as exc:
		pytest.skip(f"dev_cfg.json does not load with this qblox-scheduler: {exc}")
	return hw, dev


@pytest.fixture(scope="module")
def sim(configs):
	hw, dev = configs
	return SimulatedCluster(hw, dev, seed=5)


def _ghz(n=3):
	qc = QuantumCircuit(n, n)
	qc.h(0)
	for k in range(1, n):
		qc.cx(0, k)
	qc.measure(range(n), range(n))
	return qc


def _probs(sim, qc):
	schedule, _ = uq.build_schedule(qc, 1)
	channels, probs = sim.probabilities(schedule)
	return {format(i, f"0{len(channels)}b"): float(p) for i, p in enumerate(probs.ravel())}


def _spans(series):
	on = np.nonzero(np.abs(series) > 0)[0]
	breaks = np.nonzero(np.diff(on) > 1)[0]
	starts = np.concatenate([[on[0]], on[breaks + 1]])
	stops = np.concatenate([on[breaks], [on[-1]]])
	return list(zip(starts, stops))


# --- Q1ASM -> pulses ----------------------------------------------------------

def test_q1asm_renders_the_ghz_pulses(configs):
	hw, dev = configs
	schedule, _ = uq.build_schedule(_ghz(), 1)
	timeline = q1run.run_compiled(derive.make_agent(hw, dev).compile(schedule), dev, hw)

	# H on q0 is Y90 then X180: envelope phases 90 and 0 degrees
	pulses = _spans(timeline.drive["q0"])
	phases = [round(float(np.degrees(np.angle(timeline.drive["q0"][(a + b) // 2])))) for a, b in pulses]
	assert phases == [90, 0]
	# both CZs pulse q0's flux line, back to back: 2 x 40 ns
	(a, b), = _spans(timeline.flux["q0"])
	assert b - a + 1 == 80
	# all three readouts start together; windows follow dev_cfg
	assert sorted(timeline.acquisitions) == [0, 1, 2]
	starts = {s for s, _ in timeline.acquisitions.values()}
	assert len(starts) == 1
	assert {c: e - s for c, (s, e) in timeline.acquisitions.items()} == {0: 5000, 1: 1900, 2: 5000}


# --- the simulated chip -------------------------------------------------------

def test_sim_device_round_trip_and_checks():
	data = {"qubits": {"q0": {"t1_us": 20, "t2_us": 10}}, "couplings": {}}
	device = SimDevice.from_dict(data)
	assert SimDevice.from_dict(device.to_dict()).to_dict() == device.to_dict()
	for bad in ({"qubits": {"q0": {"t1_us": 10, "t2_us": 30}}},
		    {"qubits": {"q0": {"t1": 10}}},
		    {"qubits": {"q0": {}}, "couplings": {"q0_q9": {"g_mhz": 5}}},
		    {"qubits": {"q0": {}}, "readout_decay_fraction": 2}):
		with pytest.raises(SimDeviceError):
			SimDevice.from_dict(bad)


def test_matched_chip_and_calibrated_config(sim):
	cz = sim.device_cfg["edges"]
	# the second CZ on q0's flux line moves to 3x amplitude
	assert cz["q0_q2"]["cz"]["square_amp"] == pytest.approx(3 * cz["q0_q1"]["cz"]["square_amp"])
	assert any("shares q0's flux line" in note for note in sim.notes)
	# DRAG beta lands on 1 / (2 |anharmonicity|)
	expected = 1 / (2 * 2 * np.pi * 0.2e9)
	for element in sim.device_cfg["elements"].values():
		assert element["rxy"]["beta"] == pytest.approx(expected, rel=0.03)
	# CZs: conditional phase near 180 degrees, little leakage
	for note in (n for n in sim.notes if "conditional phase" in n):
		phase = float(note.split("conditional phase ")[1].split(" deg")[0])
		leak = float(note.split("leakage ")[1].split("%")[0])
		assert abs(phase - 180) < 5 and leak < 1


def test_phase_corrections_undo_the_cz_phases(configs, sim):
	from qblox_scheduler.operations import CZ, Rxy

	hw, _ = configs

	def ramsey(cfg):
		def build(schedule):
			schedule.add(Rxy(theta=90.0, phi=0.0, qubit="q0"))
			schedule.add(CZ("q0", "q2"))
			schedule.add(Rxy(theta=-90.0, phi=0.0, qubit="q0"))
		timeline = derive._render(hw, cfg, build)
		model = engine.Model(sim.device, ["q0", "q2"])
		active = timeline.active_ns()
		psi = model.evolve(timeline, model.basis({}), int(active[0]) - 2, int(active[-1]) + 3,
				   decoherence=False)
		return float(1 - model.populations(psi)[0].sum())

	flipped = copy.deepcopy(sim.device_cfg)
	for edge in flipped["edges"].values():
		edge["cz"]["parent_phase_correction"] *= -1
		edge["cz"]["child_phase_correction"] *= -1
	assert ramsey(sim.device_cfg) < 0.002
	assert ramsey(flipped) > 5 * ramsey(sim.device_cfg)


def test_gate_probabilities(sim):
	qc = QuantumCircuit(1, 1)
	qc.x(0)
	qc.measure(0, 0)
	assert 0.85 < _probs(sim, qc)["1"] < 0.97          # readout decay pulls 1 toward 0
	qc = QuantumCircuit(1, 1)
	qc.h(0)
	qc.measure(0, 0)
	assert 0.45 < _probs(sim, qc)["0"] < 0.6
	qc = QuantumCircuit(2, 2)
	qc.h(0)
	qc.cx(0, 1)
	qc.measure([0, 1], [0, 1])
	p = _probs(sim, qc)
	assert p["00"] + p["11"] > 0.85
	p = _probs(sim, _ghz())
	assert p["000"] + p["111"] > 0.8


def test_readout_decay_fraction(sim):
	qc = QuantumCircuit(1, 1)
	qc.x(0)
	qc.measure(0, 0)
	before = sim.device.readout_decay_fraction
	try:
		sim.device.readout_decay_fraction = 0.0
		instant = _probs(sim, qc)["1"]
		sim.device.readout_decay_fraction = 1.0
		full = _probs(sim, qc)["1"]
	finally:
		sim.device.readout_decay_fraction = before
	# q0 integrates for 5 us with T1 = 40 us
	assert full / instant == pytest.approx(np.exp(-5 / 40), rel=0.02)


def test_a_detuned_qubit_gives_worse_gates(configs, sim):
	hw, _ = configs
	chip = SimDevice.from_dict(sim.device.to_dict())
	chip.qubits["q0"].drive_detuning_ghz = 0.02
	detuned = SimulatedCluster(hw, sim.device_cfg, sim_device=chip, seed=5)
	qc = QuantumCircuit(1, 1)
	qc.sx(0)
	qc.sx(0)
	qc.measure(0, 0)
	assert _probs(detuned, qc)["1"] < _probs(sim, qc)["1"] - 0.03


# --- the service ----------------------------------------------------------------

def test_simulation_is_requested_by_env_or_service_name(monkeypatch):
	monkeypatch.delenv("QBLOX_SIMULATE", raising=False)
	monkeypatch.delenv("QFW_QPM_SERVICE_NAME", raising=False)
	assert not uq.simulation_requested()
	monkeypatch.setenv("QFW_QPM_SERVICE_NAME", "qblox-sim")
	assert uq.simulation_requested()
	monkeypatch.setenv("QBLOX_SIMULATE", "0")
	assert not uq.simulation_requested()
	monkeypatch.setenv("QBLOX_SIMULATE", "maybe")
	with pytest.raises(Exception, match="QBLOX_SIMULATE"):
		uq.simulation_requested()


def test_service_runs_on_the_simulated_cluster(monkeypatch, tmp_path, sim):
	from util.circuit_payload import encode_qiskit_circuit, qiskit_circuit_formats

	monkeypatch.setenv("QBLOX_SIMULATE", "1")
	monkeypatch.setenv("QBLOX_SIM_SEED", "3")
	monkeypatch.delenv("QBLOX_CLUSTER_IP", raising=False)
	monkeypatch.chdir(tmp_path)
	sim_path = tmp_path / "sim_device.json"
	cfg_path = tmp_path / "dev_cfg.sim.json"
	sim.device.save(sim_path)
	cfg_path.write_text(json.dumps(sim.device_cfg))
	monkeypatch.setenv("QBLOX_SIM_DEVICE", str(sim_path))
	monkeypatch.setenv("QBLOX_DEVICE_CONFIG", str(cfg_path))

	class Circ:
		provider_credential = None

		def __init__(self, info):
			self.info = info

		def get_cid(self):
			return "c1"

	client = uq.QbloxServiceClient()

	def run(qc, shots=2000):
		info = dict(encode_qiskit_circuit(qc, qiskit_circuit_formats()))
		info["num_shots"] = shots
		return client.run_circuit(Circ(info))

	qc = QuantumCircuit(3, 3)
	qc.x(0)
	qc.measure([0, 1, 2], [0, 1, 2])
	out = run(qc)
	assert max(out["counts"], key=out["counts"].get) == "001"
	assert out["qblox"]["mode"] == "simulated"
	counts = run(_ghz())["counts"]
	assert sum(counts.values()) == 2000
	assert (counts.get("000", 0) + counts.get("111", 0)) / 2000 > 0.8
	assert client.get_backend_info()["mode"] == "simulated"
