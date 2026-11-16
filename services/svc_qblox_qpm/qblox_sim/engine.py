"""
QuTiP model of the simulated chip, driven by the Timeline that q1run renders.

Each qubit is a three-level transmon in the frame of its drive clock:

	H_q(t) = 2pi*det_q*n + pi*alpha_q*n(n-1) + delta_q(t)*n
	         + (Omega_q(t)*a^dag + conj(Omega_q(t))*a) / 2
	Omega_q(t) = drive_rad_per_ns_per_v * E_q(t)       (E: complex envelope)
	delta_q(t) = 2pi * flux_ghz_per_v * F_q(t)          (F: flux line)

and each coupling is an exchange term in the two clock frames:

	J * (a_p^dag a_c * exp(i*D*t) + h.c.),   D = 2pi*(clock_p - clock_c)

T1 and T2 enter as collapse operators (a for decay, n for pure dephasing).

Exchange terms are kept only while a flux pulse plays on one of the pair:
off resonance they average out, so idle ZZ and swaps are left out. Time is
in ns throughout; 0 is the sequencers' sync.
"""
import itertools

import numpy as np

TWO_PI = 2 * np.pi
LEVELS = 3
COUPLING_MARGIN_NS = 2
MIN_IDLE_NS = 4


class SimulationError(RuntimeError):
	pass


def _runs(labels):
	"""Run-length encode a sequence: [(label, start, stop), ...]."""
	runs = []
	start = 0
	for i in range(1, len(labels) + 1):
		if i == len(labels) or labels[i] != labels[start]:
			runs.append((labels[start], start, i))
			start = i
	return runs


class Model:
	def __init__(self, device, elements, levels=LEVELS):
		import qutip as qt

		missing = [e for e in elements if e not in device.qubits]
		if missing:
			raise SimulationError(f"the simulated chip has no qubits {missing}")
		self.qt = qt
		self.device = device
		self.elements = list(elements)
		self.levels = levels
		self.index = {e: i for i, e in enumerate(self.elements)}
		n = len(self.elements)

		def on(op, k):
			return qt.tensor([op if i == k else qt.qeye(levels) for i in range(n)])

		a = qt.destroy(levels)
		num = qt.num(levels)
		self.a = {e: on(a, k) for e, k in self.index.items()}
		self.n = {e: on(num, k) for e, k in self.index.items()}

		self.H0 = 0 * self.n[self.elements[0]]
		self.c_ops = []
		for e in self.elements:
			q = device.qubits[e]
			self.H0 += TWO_PI * q.drive_detuning_ghz * self.n[e]
			self.H0 += np.pi * q.anharmonicity_ghz * self.n[e] * (self.n[e] - 1)
			t1 = q.t1_us * 1e3
			t2 = q.t2_us * 1e3
			self.c_ops.append(np.sqrt(1 / t1) * self.a[e])
			gamma_phi = max(1 / t2 - 1 / (2 * t1), 0.0)
			if gamma_phi > 0:
				self.c_ops.append(np.sqrt(2 * gamma_phi) * self.n[e])

		self.edges = []
		for name, spec in device.couplings.items():
			parent, _, child = name.partition("_")
			if parent in self.index and child in self.index:
				qp, qc = device.qubits[parent], device.qubits[child]
				clock_p = qp.frequency_ghz - qp.drive_detuning_ghz
				clock_c = qc.frequency_ghz - qc.drive_detuning_ghz
				ap, ac = self.a[parent], self.a[child]
				self.edges.append({
					"name": name, "parent": parent, "child": child,
					"J": TWO_PI * float(spec["g_mhz"]) / 1e3,
					"D": TWO_PI * (clock_p - clock_c),
					"xx": ap.dag() * ac + ap * ac.dag(),
					"yy": 1j * (ap.dag() * ac - ap * ac.dag()),
				})

	# ------------------------------------------------------------- states
	def ground(self):
		psi = self.qt.tensor([self.qt.basis(self.levels, 0) for _ in self.elements])
		return psi * psi.dag()

	def basis(self, levels_by_element):
		kets = [self.qt.basis(self.levels, levels_by_element.get(e, 0)) for e in self.elements]
		return self.qt.tensor(kets)

	# --------------------------------------------------------- evolution
	def _segments(self, timeline, t0, t1):
		"""Classify each ns in [t0, t1) as idle, driven or coupled."""
		length = t1 - t0
		driven = np.zeros(length, dtype=bool)
		flux_on = {}
		for e in self.elements:
			for store in (timeline.drive, timeline.flux):
				series = store.get(e)
				if series is not None:
					chunk = np.abs(series[t0:t1]) > 0
					driven[:len(chunk)] |= chunk
			series = timeline.flux.get(e)
			on = np.zeros(length, dtype=bool)
			if series is not None:
				chunk = np.abs(series[t0:t1]) > 0
				on[:len(chunk)] = chunk
			flux_on[e] = on
		coupled = np.zeros(length, dtype=bool)
		for edge in self.edges:
			active = flux_on[edge["parent"]] | flux_on[edge["child"]]
			idx = np.nonzero(active)[0]
			for k in idx:
				lo = max(k - COUPLING_MARGIN_NS, 0)
				hi = min(k + COUPLING_MARGIN_NS + 1, length)
				coupled[lo:hi] = True
		labels = np.where(coupled, 2, np.where(driven, 1, 0))
		runs = _runs(list(labels))
		# fold short idle gaps into the driven segment around them
		merged = []
		for label, a, b in runs:
			if label == 0 and b - a < MIN_IDLE_NS:
				label = 1
			if merged and merged[-1][0] == label:
				merged[-1] = (label, merged[-1][1], b)
			else:
				merged.append((label, a, b))
		return [(label, a + t0, b + t0) for label, a, b in merged]

	def _hamiltonian(self, timeline, a, b, coupled):
		qt = self.qt
		times = np.arange(a, b + 1, dtype=float)
		terms = [self.H0]

		def window(series):
			out = np.zeros(len(times), dtype=series.dtype)
			chunk = series[a:b + 1]
			out[:len(chunk)] = chunk
			return out

		for e in self.elements:
			q = self.device.qubits[e]
			series = timeline.drive.get(e)
			if series is not None and np.any(series[a:b]):
				omega = q.drive_rad_per_ns_per_v * window(series)
				ad = self.a[e].dag()
				terms.append([self.a[e] + ad, qt.coefficient(omega.real / 2, tlist=times, order=0)])
				terms.append([1j * (ad - self.a[e]), qt.coefficient(omega.imag / 2, tlist=times, order=0)])
			series = timeline.flux.get(e)
			if series is not None and np.any(series[a:b]):
				delta = TWO_PI * q.flux_ghz_per_v * window(series)
				terms.append([self.n[e], qt.coefficient(delta, tlist=times, order=0)])
		max_step = 0.25
		if coupled:
			for edge in self.edges:
				fp = timeline.flux.get(edge["parent"])
				fc = timeline.flux.get(edge["child"])
				lo, hi = max(a - COUPLING_MARGIN_NS, 0), b + COUPLING_MARGIN_NS
				if not ((fp is not None and np.any(fp[lo:hi])) or (fc is not None and np.any(fc[lo:hi]))):
					continue
				J, D = edge["J"], edge["D"]
				terms.append([edge["xx"], qt.coefficient(
					lambda t, J=J, D=D: J * np.cos(D * t))])
				terms.append([edge["yy"], qt.coefficient(
					lambda t, J=J, D=D: J * np.sin(D * t))])
				if D:
					max_step = min(max_step, 0.1 * TWO_PI / abs(D))
		if len(terms) == 1:
			return self.H0, max_step
		return qt.QobjEvo(terms), max_step

	def evolve(self, timeline, state, t0, t1, decoherence=True):
		"""Evolve a density matrix (or ket, without decoherence) from t0 to t1."""
		qt = self.qt
		c_ops = self.c_ops if decoherence else []
		for label, a, b in self._segments(timeline, t0, t1):
			if b <= a:
				continue
			if label == 0:
				H, max_step = self.H0, float(b - a)
			else:
				H, max_step = self._hamiltonian(timeline, a, b, coupled=(label == 2))
			options = {"max_step": max_step, "atol": 1e-9, "rtol": 1e-7, "nsteps": 10**6,
				   "progress_bar": False, "store_final_state": True,
				   "store_states": False}
			if c_ops:
				result = qt.mesolve(H, state, [float(a), float(b)], c_ops=c_ops, options=options)
			else:
				result = qt.sesolve(H, state, [float(a), float(b)], options=options)
			state = result.final_state
		return state

	# ------------------------------------------------------- measurement
	def populations(self, rho):
		probs = np.real(np.diag(rho.full())) if rho.isoper else np.abs(rho.full().ravel()) ** 2
		probs = np.clip(probs, 0, None)
		return (probs / probs.sum()).reshape([self.levels] * len(self.elements))

	def decay_populations(self, pops, element, duration_ns):
		"""Amplitude damping of one qubit's level populations over a time."""
		if duration_ns <= 0:
			return pops
		from scipy.linalg import expm
		gamma = 1 / (self.device.qubits[element].t1_us * 1e3)
		G = np.zeros((self.levels, self.levels))
		for level in range(1, self.levels):
			G[level, level] = -level * gamma
			G[level - 1, level] = level * gamma
		M = expm(G * duration_ns)
		axis = self.index[element]
		return np.moveaxis(np.tensordot(M, np.moveaxis(pops, axis, 0), axes=([1], [0])), 0, axis)


def readout_distribution(model, pops, measured):
	"""Joint probability of bits for the measured elements, in order.

	Level 0 reads as 0; levels 1 and 2 read as 1; then each qubit's assignment
	errors apply.
	"""
	keep = [model.index[e] for e in measured]
	drop = tuple(i for i in range(len(model.elements)) if i not in keep)
	marg = pops.sum(axis=drop) if drop else pops
	# sum() leaves the kept axes in index order; reorder them to `measured`
	remaining = sorted(keep)
	marg = np.transpose(marg, [remaining.index(i) for i in keep])
	bit_probs = np.zeros([2] * len(measured))
	for levels in itertools.product(range(model.levels), repeat=len(measured)):
		p = marg[levels]
		if p == 0:
			continue
		ideal = tuple(0 if lvl == 0 else 1 for lvl in levels)
		bit_probs[ideal] += p
	for k, element in enumerate(measured):
		q = model.device.qubits[element]
		C = np.array([[1 - q.readout_p01, q.readout_p10],
			      [q.readout_p01, 1 - q.readout_p10]])  # C[read, true]
		bit_probs = np.moveaxis(np.tensordot(C, np.moveaxis(bit_probs, k, 0), axes=([1], [0])), 0, k)
	return bit_probs
