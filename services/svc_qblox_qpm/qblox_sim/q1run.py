"""
Run a compiled Qblox schedule's Q1ASM programs in q1simulator and turn the
rendered outputs into what each qubit sees.

Each compiled sequencer program runs unchanged. Only the simulated
instrument is configured for simulation: every sequencer renders I and Q at
an NCO frequency of 0 with its phase registers still applied, so its two
outputs are the complex drive envelope in the frame of its clock, including
every reset_ph, set_ph and set_ph_delta the program issues. That is the drive
a qubit sees after ideal up-conversion, whatever the real module's mode,
modulation frequency or LO.

The result is a Timeline on a 1 ns grid, time 0 at the sequencers' sync:
	drive[element]   complex envelope (volts at the simulated output)
	flux[element]    flux line amplitude (volts)
	acquisitions     acq_channel -> (start_ns, stop_ns)
"""
import time
from dataclasses import dataclass, field

import numpy as np

DEFAULT_INTEGRATION_NS = 1000


class Q1RunError(RuntimeError):
	pass


@dataclass
class Timeline:
	length_ns: int
	drive: dict = field(default_factory=dict)
	flux: dict = field(default_factory=dict)
	acquisitions: dict = field(default_factory=dict)
	programs: dict = field(default_factory=dict)

	def active_ns(self):
		"""Sample indices where any drive or flux line is non-zero."""
		active = np.zeros(self.length_ns, dtype=bool)
		for series in list(self.drive.values()) + list(self.flux.values()):
			active[:len(series)] |= np.abs(series) > 0
		return np.nonzero(active)[0]


def _connectivity(hardware_cfg):
	"""Instrument channel ("cluster0.module2.real_output_0") -> ports."""
	ports = {}
	for a, b in (hardware_cfg.get("connectivity") or {}).get("graph", []):
		ports.setdefault(a, []).append(b)
	return ports


def _element_ports(device_cfg):
	"""port -> (element, kind) for the microwave and flux ports."""
	out = {}
	for name, element in (device_cfg.get("elements") or {}).items():
		ports = element.get("ports") or {}
		for kind in ("microwave", "flux"):
			if ports.get(kind):
				out[ports[kind]] = (name, kind)
	return out


def _acq_channels(device_cfg):
	"""acq_channel -> element name."""
	out = {}
	for name, element in (device_cfg.get("elements") or {}).items():
		channel = (element.get("measure") or {}).get("acq_channel")
		if channel is not None:
			out[int(channel)] = name
	return out


def _acq_mapping(module_settings, seq_name):
	"""acq_channel -> hardware acquisition index, from qblox-scheduler's
	acq_hardware_mapping (None when the compiled schedule has none)."""
	mapping = (module_settings.get("acq_hardware_mapping")
		   if isinstance(module_settings, dict) else
		   getattr(module_settings, "acq_hardware_mapping", None))
	if not mapping or seq_name not in mapping:
		return None
	entry = mapping[seq_name]
	out = {}
	for channel, bins in (getattr(entry, "non_binned_and_append", None) or {}).items():
		for spec in bins.values():
			out[int(channel)] = int(spec.index)
	return out


def _sequencer_items(compiled_instructions):
	"""(cluster, module key, slot, sequencer index, settings, acq map) per
	sequencer."""
	for cluster, modules in compiled_instructions.items():
		if not isinstance(modules, dict):
			continue
		for module_key, settings in modules.items():
			if not module_key.startswith(cluster + "_module"):
				continue
			seqs = getattr(settings, "sequencers", None)
			if seqs is None and isinstance(settings, dict):
				seqs = settings.get("sequencers")
			for name, seq in (seqs or {}).items():
				slot = int(module_key[len(cluster + "_module"):])
				yield (cluster, module_key, slot, int(name.replace("seq", "")), seq,
				       _acq_mapping(settings, name))


def _attr(seq, name, default=None):
	if isinstance(seq, dict):
		return seq.get(name, default)
	return getattr(seq, name, default)


def run_compiled(compiled_schedule, device_cfg, hardware_cfg, timeout_s=120):
	"""Execute every sequencer program and return a Timeline."""
	import logging
	try:
		from q1simulator import Cluster as SimCluster
	except Exception as exc:  # qtpy raises its own error when no Qt binding is installed
		raise Q1RunError(
			f"q1simulator does not import ({exc}); it needs Python 3.12 or later "
			"and a Qt binding such as PyQt5, which qblox-scheduler 1.0.0b4 brings") from exc

	instructions = compiled_schedule.compiled_instructions
	items = list(_sequencer_items(instructions))
	if not items:
		raise Q1RunError(
			"the compiled schedule has no sequencer programs; was it "
			"compiled with a hardware config?")

	connectivity = _connectivity(hardware_cfg)
	element_ports = _element_ports(device_cfg)
	acq_elements = _acq_channels(device_cfg)

	slots = {}
	for _, _, slot, _, seq, _ in items:
		is_qrm = bool(_attr(seq, "connected_input_indices", ()))
		slots[slot] = "QRM" if (is_qrm or slots.get(slot) == "QRM") else "QCM"

	previous = logging.root.manager.disable
	logging.disable(logging.WARNING)  # q1simulator logs every unloaded sequencer
	try:
		sim = SimCluster("qfw_sim", {slot: kind for slot, kind in slots.items()})
		roles = []
		for cluster, module_key, slot, index, seq, acq_map in items:
			sequencer = getattr(sim, f"module{slot}").sequencers[index]
			for parameter, value in (
					("sync_en", True), ("mod_en_awg", True), ("nco_freq", 0.0),
					("gain_awg_path0", 1.0), ("gain_awg_path1", 1.0),
					("offset_awg_path0", 0.0), ("offset_awg_path1", 0.0)):
				if parameter in sequencer.parameters:
					sequencer.parameters[parameter](value)
			sequencer.parameters["connect_out0"]("I")
			sequencer.parameters["connect_out1"]("Q")
			sequence = _attr(seq, "sequence")
			if slots[slot] == "QRM":
				length = _attr(seq, "integration_length_acq") or DEFAULT_INTEGRATION_NS
				for parameter, value in (("demod_en_acq", False),
							 ("integration_length_acq", int(length))):
					if parameter in sequencer.parameters:
						sequencer.parameters[parameter](value)
			sequencer.sequence(sequence)

			channel = f"{cluster}.{module_key[len(cluster) + 1:]}.{_attr(seq, 'channel_name')}"
			roles.append((sequencer, sequence, channel, slots[slot] == "QRM",
				      f"{module_key}.seq{index}", acq_map))

		sim.arm_sequencer()
		sim.start_sequencer()
		# Each sequencer runs in its own thread; wait for all of them.
		deadline = time.monotonic() + timeout_s
		while any(r[0].run_state == "RUNNING" for r in roles):
			if time.monotonic() > deadline:
				for r in roles:
					r[0].stop_sequencer()
				raise Q1RunError(f"q1simulator did not finish within {timeout_s} s")
			time.sleep(0.005)
	finally:
		logging.disable(previous)
	for sequencer, _, _, _, label, _ in roles:
		status = sequencer.get_sequencer_status()
		flags = [str(f) for f in getattr(status, "err_flags", []) or []]
		if flags:
			raise Q1RunError(f"{label} stopped with errors: {', '.join(flags)}")

	outputs = []
	length = 0
	for sequencer, sequence, channel, is_qrm, label, acq_map in roles:
		out = sequencer.get_output()
		i_path = np.asarray(out["I"].data, dtype=float) if "I" in out else np.zeros(1)
		q_path = np.asarray(out["Q"].data, dtype=float) if "Q" in out else np.zeros(1)
		n = max(len(i_path), len(q_path))
		envelope = np.zeros(n, dtype=complex)
		envelope[:len(i_path)] += i_path
		envelope[:len(q_path)] += 1j * q_path
		length = max(length, n)
		outputs.append((sequencer, sequence, channel, is_qrm, label, acq_map, envelope))

	timeline = Timeline(length_ns=length)
	for sequencer, sequence, channel, is_qrm, label, acq_map, envelope in outputs:
		timeline.programs[label] = sequence.get("program", "")
		if is_qrm:
			if acq_map is not None:
				index_to_channel = {index: channel_id for channel_id, index in acq_map.items()}
			else:
				index_to_channel = {
					int(spec["index"]): int(name)
					for name, spec in (sequence.get("acquisitions") or {}).items()}
			for acq_index, times in sequencer.rt_renderer.acq_times.items():
				if not times:
					continue
				if int(acq_index) not in index_to_channel and acq_map is not None:
					continue
				channel_id = index_to_channel.get(int(acq_index), int(acq_index))
				start = int(times[0][0])
				stop = start + int(sequencer.rt_renderer.acq_conf.length)
				timeline.acquisitions[channel_id] = (start, stop)
			continue
		ports = connectivity.get(channel, [])
		targets = [element_ports[p] for p in ports if p in element_ports]
		if not targets:
			if np.any(envelope):
				raise Q1RunError(
					f"{label} plays on {channel}, which no device element's "
					f"microwave or flux port is connected to")
			continue
		for element, kind in targets:
			padded = np.zeros(length, dtype=complex)
			padded[:len(envelope)] = envelope
			if kind == "flux":
				store = timeline.flux
				padded = padded.real
			else:
				store = timeline.drive
			if element in store:
				store[element] = store[element] + padded
			else:
				store[element] = padded

	unknown = [ch for ch in timeline.acquisitions if ch not in acq_elements]
	if unknown:
		raise Q1RunError(f"acquisition channels {unknown} belong to no device element")
	return timeline
