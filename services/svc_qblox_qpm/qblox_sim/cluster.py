"""
A simulated Qblox Cluster: compile a schedule with qblox-scheduler, run its
Q1ASM in q1simulator, drive a QuTiP model of the chip with the result, and
return thresholded acquisitions shaped like HardwareAgent.run()'s dataset.

Every shot of a schedule runs the same pulses, so one repetition is rendered
and simulated, and the shots are sampled from the outcome probabilities.
"""
import copy
import time

import numpy as np

from . import derive, engine, q1run
from .physics import SimDevice


class SimulatedCluster:
	def __init__(self, hardware_cfg, device_cfg, sim_device=None, seed=None):
		"""With sim_device None, the chip is matched to device_cfg and the
		config is calibrated against it (see derive.matched)."""
		self.hardware_cfg = copy.deepcopy(hardware_cfg)
		if sim_device is None:
			sim_device, device_cfg, self.notes = derive.matched(device_cfg, hardware_cfg)
			self.matched = True
		else:
			if not isinstance(sim_device, SimDevice):
				sim_device = SimDevice.from_dict(sim_device)
			self.notes = []
			self.matched = False
		self.device = sim_device
		self.device_cfg = copy.deepcopy(device_cfg)
		self.agent = derive.make_agent(self.hardware_cfg, self.device_cfg)
		self.acq_elements = q1run._acq_channels(self.device_cfg)
		self.rng = np.random.default_rng(seed)
		self.last_run = {}

	def _elements(self, timeline, measured):
		active = set(measured)
		for store in (timeline.drive, timeline.flux):
			active |= {e for e, series in store.items() if np.any(series)}
		for name in self.device.couplings:
			parent, _, child = name.partition("_")
			flux = timeline.flux
			if any(e in flux and np.any(flux[e]) for e in (parent, child)):
				active |= {parent, child}
		order = list(self.device.qubits)
		return [e for e in order if e in active]

	def probabilities(self, schedule):
		"""Compile and simulate one repetition: (channels, bit probabilities)."""
		started = time.monotonic()
		compiled = self.agent.compile(schedule)
		compiled_at = time.monotonic()
		timeline = q1run.run_compiled(compiled, self.device_cfg, self.hardware_cfg)
		rendered_at = time.monotonic()
		if not timeline.acquisitions:
			raise engine.SimulationError("the schedule acquires nothing")
		channels = sorted(timeline.acquisitions)
		measured = [self.acq_elements[c] for c in channels]
		model = engine.Model(self.device, self._elements(timeline, measured))

		t_meas = min(timeline.acquisitions[c][0] for c in channels)
		active = timeline.active_ns()
		active = active[active < t_meas]
		t_first = int(active[0]) if len(active) else t_meas
		rho = model.evolve(timeline, model.ground(), t_first, t_meas)
		pops = model.populations(rho)
		fraction = self.device.readout_decay_fraction
		for channel, element in zip(channels, measured):
			start, stop = timeline.acquisitions[channel]
			pops = model.decay_populations(pops, element, (start - t_meas) + fraction * (stop - start))
		probs = engine.readout_distribution(model, pops, measured)
		self.last_run = {
			"timeline": timeline, "elements": model.elements,
			"compile_s": compiled_at - started, "q1asm_s": rendered_at - compiled_at,
			"physics_s": time.monotonic() - rendered_at,
		}
		return channels, probs

	def run(self, schedule, shots):
		"""Thresholded acquisitions for `shots` repetitions, as an xarray
		Dataset keyed by acq_channel: +1 reads 1, -1 reads 0."""
		import xarray as xr

		channels, probs = self.probabilities(schedule)
		flat = probs.ravel()
		flat = flat / flat.sum()
		outcomes = self.rng.choice(len(flat), size=int(shots), p=flat)
		bits = np.array(np.unravel_index(outcomes, probs.shape))  # (channels, shots)
		data = {channel: ("repetition", np.where(bits[k] == 1, 1.0, -1.0))
			for k, channel in enumerate(channels)}
		return xr.Dataset(data)
