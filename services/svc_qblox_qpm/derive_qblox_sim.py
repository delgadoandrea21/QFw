#!/usr/bin/env python3
"""
Write the simulated chip the qblox-sim service would match to a Qblox device
config, and the config calibrated against it.

The service does this itself at start-up when QBLOX_SIM_DEVICE is unset.
Writing the files once makes start-up instant, lets you read what the
simulated chip is, and lets you change it: shorten a T1, detune a qubit from
its drive clock, or put back the uncalibrated config to see what that costs.
Point the service at both files together:

	QBLOX_SIM_DEVICE=<out-device>  QBLOX_DEVICE_CONFIG=<out-config>

Needs qblox-scheduler (1.0.0b4, for BasicTransmonElement device configs),
q1simulator and qutip.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from qblox_sim import derive  # noqa: E402
from qblox_sim.physics import SimDevice  # noqa: E402


def main(argv=None):
	parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
	parser.add_argument("--device", default=os.path.join(HERE, "configs", "dev_cfg.json"),
			    help="Qblox device config (default: the service's configs/dev_cfg.json)")
	parser.add_argument("--hardware", default=os.path.join(HERE, "configs", "hw_cfg.json"),
			    help="Qblox hardware config (default: configs/hw_cfg.json)")
	parser.add_argument("--base", help="simulated-chip JSON whose T1, T2, anharmonicity "
			    "and readout errors to keep")
	parser.add_argument("--out-device", default="sim_device.json",
			    help="where to write the simulated chip (default: sim_device.json)")
	parser.add_argument("--out-config", help="where to write the calibrated device config "
			    "(default: <device>.sim.json in the current directory)")
	parser.add_argument("--dry-run", action="store_true", help="print, write nothing")
	args = parser.parse_args(argv)

	with open(args.device) as handle:
		device_cfg = json.load(handle)
	with open(args.hardware) as handle:
		hardware_cfg = json.load(handle)
	base = SimDevice.load(args.base) if args.base else None

	chip, calibrated, notes = derive.matched(device_cfg, hardware_cfg, base=base)

	print(f"{'qubit':6} {'GHz':>7} {'anharm':>7} {'T1 us':>6} {'T2 us':>6} "
	      f"{'drive rad/ns/V':>15} {'flux GHz/V':>11} {'DRAG beta ns':>13}")
	for name, q in chip.qubits.items():
		beta = calibrated["elements"].get(name, {}).get("rxy", {}).get("beta")
		print(f"{name:6} {q.frequency_ghz:7.3f} {q.anharmonicity_ghz:7.3f} {q.t1_us:6.1f} "
		      f"{q.t2_us:6.1f} {q.drive_rad_per_ns_per_v:15.4f} {q.flux_ghz_per_v:11.4f} "
		      f"{(beta or 0) * 1e9:13.3f}")
	for name, spec in chip.couplings.items():
		print(f"coupling {name}: {spec['g_mhz']:.2f} MHz")
	for note in notes:
		print(f"note: {note}")
	if args.dry_run:
		return 0

	out_config = args.out_config or (
		os.path.splitext(os.path.basename(args.device))[0] + ".sim.json")
	chip.save(args.out_device)
	with open(out_config, "w") as handle:
		json.dump(calibrated, handle, indent=2)
		handle.write("\n")
	print(f"wrote {args.out_device} and {out_config}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
