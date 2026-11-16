"""A simulated Qblox Cluster for svc_qblox_qpm (the qblox-sim service).

q1run runs compiled Q1ASM in q1simulator; physics holds the simulated chip;
engine drives a QuTiP model of it with the rendered pulses; derive matches a
chip to a Qblox device config and calibrates the config against it; cluster
ties them together behind HardwareAgent.run()'s result shape.
"""
