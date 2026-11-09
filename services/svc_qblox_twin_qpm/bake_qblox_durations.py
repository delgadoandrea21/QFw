#!/usr/bin/env python3
"""Write Qblox gate durations into a copy of the twin's calibration file.

Run this where qblox-scheduler is installed (and, for transmon QuantumDevice
files, qblox_applications), such as your lab's Qblox environment, then deploy
the file it writes. The twin's QPM reads that file with QFW_QBLOX_TWIN_DEVICE
and needs no Qblox packages at run time. See "Qblox packages" in README.md.

	python bake_qblox_durations.py --device quantum_device.yaml \\
		[--hardware hardware_config.json] [--calibration device.json] \\
		[--elements q0,q1,q2,q3] [--output device.qblox.json] [--dry-run]

It compiles one Rxy, Measure and Reset per qubit and one CZ per coupler with
qblox-scheduler, then replaces each qubit's sq_duration_ns and each coupler's
cz_duration_ns. Every other number in the calibration is kept. The Measure and
Reset durations are recorded under "qblox_timing" in the output; the twin does
not use them yet (readout takes no time in the twin today).

The script needs only this directory (the qfw_qblox_twin package), numpy and
qblox-scheduler, not a QFw install.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CALIBRATION = os.path.join(HERE, "device.example.json")


def parse_args(argv):
	parser = argparse.ArgumentParser(
		description="Write Qblox gate durations into a copy of the twin's "
		"calibration file.")
	parser.add_argument(
		"--device", required=True,
		help="QuantumDevice YAML/JSON, or a DeviceCompilationConfig JSON")
	parser.add_argument(
		"--hardware",
		help="Qblox hardware config JSON; compiles down to Q1ASM as well")
	parser.add_argument(
		"--calibration", default=DEFAULT_CALIBRATION,
		help="the twin calibration to start from (default: the bundled "
		"device.example.json)")
	parser.add_argument(
		"--elements",
		help="device element for each twin qubit, comma separated "
		"(default: q0,q1,...)")
	parser.add_argument(
		"--output",
		help="where to write the result (default: <calibration name>.qblox.json "
		"in the current directory)")
	parser.add_argument(
		"--dry-run", action="store_true",
		help="print the durations without writing a file")
	return parser.parse_args(argv)


def _qblox_scheduler_version():
	try:
		import qblox_scheduler
	except ImportError:
		raise SystemExit(
			"bake_qblox_durations.py needs qblox-scheduler (tested with 1.0.1). "
			"Run it in an environment that has it: your lab's Qblox "
			"environment, or one built from setup/qblox-twin-requirements.txt.") from None
	return getattr(qblox_scheduler, "__version__", "unknown")


def _element_names(spec, qubits):
	if not spec:
		return None
	names = [name.strip() for name in spec.split(",")]
	if len(names) != len(qubits) or not all(names):
		raise SystemExit(
			f"--elements names {len(names)} elements; the calibration has "
			f"{len(qubits)} qubits")
	return {q: names[index] for index, q in enumerate(sorted(qubits))}


def _default_output(calibration):
	stem = os.path.splitext(os.path.basename(calibration))[0]
	return os.path.join(os.getcwd(), f"{stem}.qblox.json")


def main(argv=None):
	args = parse_args(argv)
	version = _qblox_scheduler_version()
	if HERE not in sys.path:
		sys.path.insert(0, HERE)
	from qfw_qblox_twin import DeviceCal
	from qfw_qblox_twin import qblox_timing

	calibration = DeviceCal.load(args.calibration)
	names = _element_names(args.elements, calibration.qubits)
	try:
		config = qblox_timing.compilation_config(args.device, args.hardware)
		durations = qblox_timing.qblox_operation_durations(
			calibration, config, element_names=names)
	except Exception as error:
		raise SystemExit(
			f"qblox-scheduler could not time the device: "
			f"{type(error).__name__}: {error}") from None
	baked = qblox_timing.apply_qblox_durations(calibration, durations)

	element = {q: (names[q] if names else f"q{q}") for q in calibration.qubits}
	print(f"Gate durations from {args.device} (qblox-scheduler {version})")
	print(f"{'qubit':<7}{'element':<14}{'pulse ns (was)':>18}{'measure ns':>13}{'reset ns':>12}")
	for q in sorted(calibration.qubits):
		was = calibration.qubits[q].sq_duration_ns
		now = baked.qubits[q].sq_duration_ns
		reset = durations.get(("reset", q))
		print(f"{q:<7}{element[q]:<14}{f'{now:g} ({was:g})':>18}"
		      f"{durations.get(('measure', q), float('nan')):>13g}"
		      f"{(reset if reset is not None else float('nan')):>12g}")
	print(f"{'coupler':<21}{'CZ ns (was)':>18}")
	for old, new in zip(calibration.couplers, baked.couplers):
		pair = "-".join(str(q) for q in old.qubits)
		print(f"{pair:<21}{f'{new.cz_duration_ns:g} ({old.cz_duration_ns:g})':>18}")

	if args.dry_run:
		print("Dry run: nothing written.")
		return 0

	data = baked.to_dict()
	data["qblox_timing"] = {
		"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
		"qblox_scheduler": version,
		"device_config": os.path.abspath(args.device),
		"hardware_config": os.path.abspath(args.hardware) if args.hardware else None,
		"source_calibration": os.path.abspath(args.calibration),
		"elements": {str(q): element[q] for q in sorted(element)},
		"measure_ns": {str(q): durations[("measure", q)]
			       for q in sorted(calibration.qubits) if ("measure", q) in durations},
		"reset_ns": {str(q): durations[("reset", q)]
			     for q in sorted(calibration.qubits) if ("reset", q) in durations},
	}
	output = args.output or _default_output(args.calibration)
	with open(output, "w", encoding="utf-8") as stream:
		json.dump(data, stream, indent=2)
		stream.write("\n")
	print(f"Wrote {output}. Deploy it and point the twin at it with "
	      f"QFW_QBLOX_TWIN_DEVICE={output}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
