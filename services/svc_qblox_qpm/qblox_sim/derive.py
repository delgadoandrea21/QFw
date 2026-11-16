"""
Build a simulated chip that matches a Qblox device config, and calibrate the
config against it.

The device config says how to drive the qubits (pulse amplitudes, CZ flux
amplitudes and lengths, phase corrections); it doesn't say what the qubits
are. matched() picks a chip on which that config works:

- drive strength: from the rendered Rxy(180) pulse of each qubit, so that
  pulse is a pi rotation;
- frequencies and flux sensitivity: each CZ's flux amplitude brings
  |11> into resonance with |20> (flux-pulsed qubit excited twice) or, for a
  second CZ on the same flux line, with |02>, so that the pair's partners
  sit far apart;
- coupling: each CZ's flux pulse lasts one full |11> -> |20> -> |11> cycle,
  which gives the conditional phase of pi.

It then calibrates a copy of the config against that chip, as a lab would:
each qubit's DRAG beta (the transmon's third level shifts and tilts plain
Gaussian pulses), and each CZ's parent_phase_correction and
child_phase_correction (the single-qubit phases the CZ leaves). When two CZs
share a flux line at the same amplitude, no single chip can resonate both;
the copy then plays the second at three times the amplitude, resonating
through |02>: every coupled pair then stays at least 300 MHz from resonance
when idle and during the other CZ. The notes say so. More than two CZs per
flux line are matched but not spread.

Everything not set by the config (T1, T2, anharmonicity, readout errors)
comes from physics.QUBIT_DEFAULTS, or from `base` when given.
"""
import copy

import numpy as np

from . import engine, q1run
from .physics import QUBIT_DEFAULTS, SimDevice, SimQubit

PARENT_FREQUENCY_GHZ = 6.0
FLUX_EXCURSION_GHZ = 0.3     # flux-pulsed qubit's shift at its smallest CZ amplitude
SHARED_LINE_SPREAD = 3.0     # second CZ on a shared flux line: this multiple of the first
BETA_SCAN_S = 1.5e-9         # DRAG beta search range, +/- seconds
SPECTATOR_MIN_GHZ = 0.15     # warn when a CZ brings two levels closer than this


def make_agent(hardware_cfg, device_cfg):
	"""A HardwareAgent for compiling only: no cluster address."""
	from qblox_scheduler import HardwareAgent

	hardware_cfg = copy.deepcopy(hardware_cfg)
	for desc in hardware_cfg.get("hardware_description", {}).values():
		if desc.get("instrument_type") == "Cluster":
			desc["ip"] = None
	return HardwareAgent(hardware_cfg, copy.deepcopy(device_cfg))


def _render(hardware_cfg, device_cfg, build):
	from qblox_scheduler import Schedule

	schedule = Schedule("qblox-sim probe", repetitions=1)
	build(schedule)
	agent = make_agent(hardware_cfg, device_cfg)
	return q1run.run_compiled(agent.compile(schedule), device_cfg, hardware_cfg)


def _drive_areas(hardware_cfg, device_cfg, qubits):
	from qblox_scheduler.operations import Rxy

	def build(schedule):
		for q in qubits:
			schedule.add(Rxy(theta=180.0, phi=0.0, qubit=q))

	timeline = _render(hardware_cfg, device_cfg, build)
	return {q: complex(np.sum(timeline.drive[q])) for q in qubits if q in timeline.drive}


def _cz_pulse(hardware_cfg, device_cfg, parent, child):
	from qblox_scheduler.operations import CZ

	timeline = _render(hardware_cfg, device_cfg, lambda s: s.add(CZ(parent, child)))
	for element, series in timeline.flux.items():
		on = np.nonzero(np.abs(series) > 0)[0]
		if len(on):
			return element, float(np.max(np.abs(series))), int(len(on)), timeline
	raise engine.SimulationError(f"CZ({parent}, {child}) plays no flux pulse")


def _cz_phases(device, timeline, parent, child):
	"""Single-qubit and conditional phases (rad) a CZ leaves, no decoherence."""
	elements = [parent, child]
	model = engine.Model(device, elements)
	flux = np.zeros(timeline.length_ns, dtype=bool)
	for series in timeline.flux.values():
		flux |= np.abs(series) > 0
	on = np.nonzero(flux)[0]
	t0 = max(int(on[0]) - 4, 0)
	t1 = int(on[-1]) + 5
	phases, populations = {}, {}
	for lp, lc in ((0, 0), (1, 0), (0, 1), (1, 1)):
		psi = model.basis({parent: lp, child: lc})
		out = model.evolve(timeline, psi, t0, t1, decoherence=False)
		amp = complex(psi.overlap(out))
		phases[(lp, lc)] = np.angle(amp)
		populations[(lp, lc)] = abs(amp) ** 2
	parent_phase = phases[(1, 0)] - phases[(0, 0)]
	child_phase = phases[(0, 1)] - phases[(0, 0)]
	conditional = phases[(1, 1)] - phases[(1, 0)] - phases[(0, 1)] + phases[(0, 0)]
	return parent_phase, child_phase, conditional, 1 - populations[(1, 1)]


PROBE_GAP_S = 200e-9


def _pulse_error(hardware_cfg, device_cfg, device, qubit):
	"""Excited population after Y90 then X180, and after X90 then Y180,
	away from the ideal 0.5: both are sensitive to a pulse's Stark shift.
	Both pairs go in one schedule, 200 ns apart, and are simulated
	separately from the ground state."""
	from qblox_scheduler.operations import IdlePulse, Rxy

	pairs = (((90.0, 90.0), (180.0, 0.0)), ((90.0, 0.0), (180.0, 90.0)))

	def build(schedule):
		for k, (first, second) in enumerate(pairs):
			if k:
				schedule.add(IdlePulse(PROBE_GAP_S))
			schedule.add(Rxy(theta=first[0], phi=first[1], qubit=qubit))
			schedule.add(Rxy(theta=second[0], phi=second[1], qubit=qubit))

	timeline = _render(hardware_cfg, device_cfg, build)
	on = np.nonzero(np.abs(timeline.drive[qubit]) > 0)[0]
	split = int(np.argmax(np.diff(on))) + 1  # the 200 ns gap
	model = engine.Model(device, [qubit])
	error = 0.0
	for span in (on[:split], on[split:]):
		psi = model.evolve(timeline, model.basis({}), max(int(span[0]) - 2, 0), int(span[-1]) + 3,
				   decoherence=False)
		error += abs(float(1 - model.populations(psi)[0]) - 0.5)
	return error


def _calibrate_beta(hardware_cfg, device_cfg, device, qubit):
	"""Scan, then refine, the DRAG beta that minimises _pulse_error."""
	cfg = copy.deepcopy(device_cfg)

	def cost(beta):
		cfg["elements"][qubit]["rxy"]["beta"] = float(beta)
		return _pulse_error(hardware_cfg, cfg, device, qubit)

	grid = np.linspace(-BETA_SCAN_S, BETA_SCAN_S, 9)
	costs = [cost(b) for b in grid]
	k = int(np.argmin(costs))
	lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
	for _ in range(8):  # golden-section refine
		m1 = hi - (hi - lo) / 1.618
		m2 = lo + (hi - lo) / 1.618
		if cost(m1) < cost(m2):
			hi = m2
		else:
			lo = m1
	beta = float((lo + hi) / 2)
	return beta, cost(beta) / 2


def _wrap_deg(rad):
	deg = np.degrees(rad)
	return float((deg + 180.0) % 360.0 - 180.0)


def matched(device_cfg, hardware_cfg, base=None):
	"""Return (SimDevice, calibrated device config, notes)."""
	elements = device_cfg.get("elements") or {}
	edges = {name: e for name, e in (device_cfg.get("edges") or {}).items()
		 if e.get("cz") and e.get("parent_element_name") and e.get("child_element_name")}
	calibrated = copy.deepcopy(device_cfg)
	notes = []

	drive_qubits = [q for q, e in elements.items() if (e.get("ports") or {}).get("microwave")]
	areas = _drive_areas(hardware_cfg, device_cfg, drive_qubits)

	qubits = {}
	for name in elements:
		spec = dict(QUBIT_DEFAULTS)
		if base is not None and name in base.qubits:
			spec.update(vars(base.qubits[name]))
		if name in areas and abs(areas[name]) > 0:
			spec["drive_rad_per_ns_per_v"] = float(np.pi / abs(areas[name]))
		qubits[name] = SimQubit(**spec)

	# CZ flux pulses, grouped by the flux line they play on
	pulses = {}
	for name, edge in edges.items():
		parent, child = edge["parent_element_name"], edge["child_element_name"]
		line, amp, length, _ = _cz_pulse(hardware_cfg, device_cfg, parent, child)
		pulses[name] = {"parent": parent, "child": child, "line": line, "amp": amp, "ns": length}

	by_line = {}
	for name, p in pulses.items():
		by_line.setdefault(p["line"], []).append(name)
	for line, names in by_line.items():
		names.sort(key=lambda n: pulses[n]["amp"])
		first = pulses[names[0]]["amp"]
		for k, name in enumerate(names[1:], start=1):
			pulses[name]["branch"] = "02" if k == 1 else "20"
			if k == 1 and pulses[name]["amp"] < 2 * first:
				scale = SHARED_LINE_SPREAD * first / pulses[name]["amp"]
				pulses[name]["amp"] *= scale
				cz = calibrated["edges"][name]["cz"]
				cz["square_amp"] = round(float(cz["square_amp"]) * scale, 12)
				notes.append(
					f"{name} shares {line}'s flux line with {names[0]} at the same "
					f"amplitude; no chip resonates both, so the calibrated config "
					f"plays {name} at square_amp {cz['square_amp']:.4g}")
		kappa = -FLUX_EXCURSION_GHZ / first if first else 0.0
		qubits[line].flux_ghz_per_v = float(kappa)
		qubits[line].frequency_ghz = PARENT_FREQUENCY_GHZ

	couplings = {}
	assigned = {}
	for name, p in pulses.items():
		line, partner = p["line"], (p["child"] if p["line"] == p["parent"] else p["parent"])
		shifted = qubits[line].frequency_ghz + qubits[line].flux_ghz_per_v * p["amp"]
		if p.get("branch") == "02":
			# |11> <-> |02>: the partner, excited twice, meets the pulsed qubit
			f_partner = shifted - qubits[partner].anharmonicity_ghz
		else:
			# |11> <-> |20>: the pulsed qubit, excited twice, meets the partner
			f_partner = shifted + qubits[line].anharmonicity_ghz
		if partner in assigned and abs(assigned[partner] - f_partner) > 1e-6:
			notes.append(f"{partner} would need two frequencies for its CZs; kept {assigned[partner]:.4f} GHz")
		else:
			qubits[partner].frequency_ghz = float(f_partner)
			assigned[partner] = f_partner
		# one full |11> -> |20> -> |11> cycle in the pulse length: sqrt(2)*J*t = pi
		g_mhz = 1e3 / (2 * np.sqrt(2) * p["ns"])
		edge = device_cfg["edges"][name]
		couplings[f"{edge['parent_element_name']}_{edge['child_element_name']}"] = {"g_mhz": float(g_mhz)}

	device = SimDevice(qubits=qubits, couplings=couplings,
			   name="matched to the Qblox device config",
			   readout_decay_fraction=(base.readout_decay_fraction if base else 0.5)).check()

	for name, p in pulses.items():
		line = p["line"]
		shifted = qubits[line].frequency_ghz + qubits[line].flux_ghz_per_v * p["amp"]
		for other in qubits:
			if other in (line, p["parent"], p["child"]) or not any(
					other in (q["parent"], q["child"]) and line in (q["parent"], q["child"])
					for q in pulses.values()):
				continue
			gap = abs(shifted - qubits[other].frequency_ghz)
			if gap < SPECTATOR_MIN_GHZ:
				notes.append(f"during {name}, {line} passes within {gap * 1e3:.0f} MHz of {other}")

	# calibrate DRAG beta on the matched chip; qubits whose pulse and
	# transmon are the same share one calibration
	done = {}
	for qubit in drive_qubits:
		rxy = calibrated["elements"][qubit]["rxy"]
		q = device.qubits[qubit]
		key = (rxy.get("amp180"), rxy.get("duration"), round(q.anharmonicity_ghz, 9),
		       round(q.drive_rad_per_ns_per_v, 9), round(q.drive_detuning_ghz, 9))
		if key not in done:
			done[key] = _calibrate_beta(hardware_cfg, calibrated, device, qubit)
		beta, error = done[key]
		rxy["beta"] = beta
		notes.append(f"{qubit}: DRAG beta {beta * 1e9:.3f} ns (Y90-X180 error {error:.2%})")

	# calibrate the phase corrections on the matched chip
	for name, edge in edges.items():
		parent, child = edge["parent_element_name"], edge["child_element_name"]
		_, _, _, timeline = _cz_pulse(hardware_cfg, calibrated, parent, child)
		p_phase, c_phase, conditional, leak = _cz_phases(device, timeline, parent, child)
		cz = calibrated["edges"][name]["cz"]
		cz["parent_phase_correction"] = _wrap_deg(p_phase)
		cz["child_phase_correction"] = _wrap_deg(c_phase)
		notes.append(
			f"{name}: conditional phase {np.degrees(conditional) % 360:.1f} deg, "
			f"|11> leakage {leak:.2%}, phase corrections "
			f"{cz['parent_phase_correction']:.1f} / {cz['child_phase_correction']:.1f} deg")
	return device, calibrated, notes
