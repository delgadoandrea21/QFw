"""The Qblox digital twin's engine and its QbloxTwinQPM facade.

The twin lives in services/svc_qblox_twin_qpm/qfw_qblox_twin and needs a real
numpy, so these tests run with the top-level suite rather than tests/mock,
whose conftest stubs numpy. The service wrapping it is tested in
tests/mock/test_qblox_twin_qpm.py.
"""
import json
import math
import pathlib
import sys
import threading

import pytest

np = pytest.importorskip("numpy")
if not hasattr(np, "random"):
    pytest.skip("numpy is the tests/mock stub here; run this file on its own",
                allow_module_level=True)

SERVICE_DIR = (pathlib.Path(__file__).resolve().parents[1] /
               "services" / "svc_qblox_twin_qpm")
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from qfw_qblox_twin import (  # noqa: E402
    DeviceCal, QubitCal, TwinBackend, hellinger_fidelity, noise)
from qfw_qblox_twin.qasm import SUPPORTED_GATES  # noqa: E402
from qfw_qblox_twin.qpm_service import QbloxTwinQPM  # noqa: E402

HDR = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
GHZ3 = HDR + "qreg q[3];\ncreg c[3];\nh q[0];\ncx q[0],q[1];\ncx q[1],q[2];\nmeasure q -> c;\n"


def ideal(n=4):
    return DeviceCal.example_4q().ideal_copy()


# ---- engine (the twin's original tests) ----------------------------------

def test_ghz_ideal():
    probs, _ = TwinBackend(ideal()).exact_probabilities(GHZ3)
    assert probs["000"] == pytest.approx(0.5, abs=1e-9)
    assert probs["111"] == pytest.approx(0.5, abs=1e-9)
    assert sum(probs.values()) == pytest.approx(1.0)


def test_ghz_noisy_is_worse_and_normalised():
    dev = DeviceCal.example_4q()
    noisy, meta = TwinBackend(dev).exact_probabilities(GHZ3)
    ref, _ = TwinBackend(dev.ideal_copy()).exact_probabilities(GHZ3)
    f = hellinger_fidelity(noisy, ref)
    assert 0.8 < f < 0.999
    assert sum(noisy.values()) == pytest.approx(1.0)
    assert meta["n_cz"] == 2


def test_bit_ordering_little_endian():
    q = HDR + "qreg q[2];\ncreg c[2];\nx q[0];\nmeasure q -> c;\n"
    probs, _ = TwinBackend(ideal()).exact_probabilities(q)
    assert probs == {"01": pytest.approx(1.0)}      # q0 -> rightmost bit


def test_cx_swap_truth():
    q = HDR + "qreg q[2];\ncreg c[2];\nx q[0];\ncx q[0],q[1];\nmeasure q -> c;\n"
    probs, _ = TwinBackend(ideal()).exact_probabilities(q)
    assert probs == {"11": pytest.approx(1.0)}
    q = HDR + "qreg q[2];\ncreg c[2];\nx q[0];\nswap q[0],q[1];\nmeasure q -> c;\n"
    probs, _ = TwinBackend(ideal()).exact_probabilities(q)
    assert probs == {"10": pytest.approx(1.0)}


def test_parameter_expressions():
    q = HDR + "qreg q[1];\ncreg c[1];\nrx(pi) q[0];\nmeasure q -> c;\n"
    probs, _ = TwinBackend(ideal()).exact_probabilities(q)
    assert probs["1"] == pytest.approx(1.0)
    q = HDR + "qreg q[1];\ncreg c[1];\nry(pi/2) q[0];\nmeasure q -> c;\n"
    probs, _ = TwinBackend(ideal()).exact_probabilities(q)
    assert probs["0"] == pytest.approx(0.5)


def test_readout_error():
    dev = ideal()
    dev.qubits[0] = QubitCal(t1_us=math.inf, t2_us=math.inf,
                             readout_p01=0.0, readout_p10=0.1, sq_error=0.0)
    q = HDR + "qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q -> c;\n"
    probs, _ = TwinBackend(dev).exact_probabilities(q)
    assert probs["0"] == pytest.approx(0.1)


def test_t1_decay_matches_exponential():
    n = 1
    rho = noise.zero_state(n)
    rho = noise.apply_1q(rho, np.array([[0, 1], [1, 0]], dtype=complex), 0, n)
    out = noise.decohere_numpy(rho, 0, n, dt_ns=30_000, t1_us=30, t2_us=30)
    p1 = np.real(out[1, 1])
    assert p1 == pytest.approx(math.exp(-1), rel=1e-9)


def test_t2_coherence_decay():
    n = 1
    plus = noise.apply_1q(noise.zero_state(n),
                          np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2), 0, n)
    out = noise.decohere_numpy(plus, 0, n, dt_ns=20_000, t1_us=40, t2_us=20)
    assert abs(out[0, 1]) / abs(plus[0, 1]) == pytest.approx(math.exp(-1), rel=1e-9)


def test_good_pair_beats_bad_pair():
    def bell(a, b):
        return (HDR + "qreg q[4];\ncreg c[2];\nh q[%d];\ncx q[%d],q[%d];\n"
                "measure q[%d] -> c[0];\nmeasure q[%d] -> c[1];\n" % (a, a, b, a, b))
    be = TwinBackend(DeviceCal.example_4q())
    ref, _ = TwinBackend(ideal()).exact_probabilities(bell(0, 1))
    good, _ = be.exact_probabilities(bell(0, 1))
    bad, _ = be.exact_probabilities(bell(2, 3))
    assert hellinger_fidelity(good, ref) > hellinger_fidelity(bad, ref)


def test_sampling_reproducible_with_seed():
    a = TwinBackend(DeviceCal.example_4q(), seed=7).run_qasm(GHZ3, 500).counts
    b = TwinBackend(DeviceCal.example_4q(), seed=7).run_qasm(GHZ3, 500).counts
    assert a == b and sum(a.values()) == 500


def test_unsupported_raises():
    with pytest.raises(NotImplementedError):
        TwinBackend(ideal()).exact_probabilities(HDR + "qreg q[1];\ncreg c[1];\nreset q[0];\n")


def test_qutip_matches_numpy():
    pytest.importorskip("qutip")
    dev = DeviceCal.example_4q()
    a, _ = TwinBackend(dev, engine="numpy").exact_probabilities(GHZ3)
    b, _ = TwinBackend(dev, engine="qutip").exact_probabilities(GHZ3)
    assert hellinger_fidelity(a, b) > 0.9999


# ---- calibration file -----------------------------------------------------

def test_example_device_file_is_example_4q():
    loaded = DeviceCal.load(SERVICE_DIR / "device.example.json")
    assert loaded.to_dict() == DeviceCal.example_4q().to_dict()
    assert json.loads((SERVICE_DIR / "device.example.json").read_text())["name"] == "example-4q"


# ---- the QPM facade the QFw service wraps -----------------------------------

def test_demo_ghz_through_the_facade():
    # examples/demo_ghz.py from the twin's own repo, as a test.
    dev = DeviceCal.load(SERVICE_DIR / "device.example.json")
    ref, _ = TwinBackend(dev.ideal_copy()).exact_probabilities(GHZ3)
    direct = TwinBackend(dev, seed=1).run_qasm(GHZ3, 4096)
    qpm = QbloxTwinQPM(str(SERVICE_DIR / "device.example.json"), seed=1)
    cid = qpm.create_circuit(GHZ3, 4096)
    assert qpm.run(cid, sync=False) == cid
    counts = qpm.get_result(cid, timeout=10)

    assert counts == direct.counts
    assert sum(counts.values()) == 4096
    f = hellinger_fidelity({k: v / 4096 for k, v in counts.items()}, ref)
    assert 0.8 < f < 0.999
    qpm.shutdown()


def test_capabilities_list_what_the_parser_accepts():
    caps = QbloxTwinQPM().capabilities()
    assert caps["num_qubits"] == 4
    assert caps["coupling_map"] == [[0, 1], [1, 2], [2, 3]]
    assert caps["basis"] == ["rz", "rx", "ry", "cz"]
    assert caps["instructions"][-1] == "measure"
    assert set(caps["instructions"]) == set(SUPPORTED_GATES) | {"measure"}
    for gate in ("rz", "rx", "ry", "cz", "cx", "h", "sx", "u3", "swap"):
        assert gate in caps["instructions"]
    assert "reset" not in caps["instructions"]


def test_sync_run_records_schedule_metadata():
    qpm = QbloxTwinQPM(seed=3)
    cid = qpm.create_circuit(GHZ3, 100)
    assert qpm.metadata(cid) == {}

    counts = qpm.run(cid, sync=True)

    assert sum(counts.values()) == 100
    meta = qpm.metadata(cid)
    assert meta["n_cz"] == 2 and meta["device"] == "example-4q"
    assert meta["duration_ns"] > 0
    qpm.delete_circuit(cid)
    assert qpm.metadata(cid) == {}
    qpm.shutdown()


def test_on_done_reports_success_and_failure():
    qpm = QbloxTwinQPM(seed=3)
    done = {}
    finished = threading.Event()

    def on_done(cid, future):
        done[cid] = future
        if len(done) == 2:
            finished.set()

    good = qpm.create_circuit(GHZ3, 50)
    bad = qpm.create_circuit(HDR + "qreg q[1];\ncreg c[1];\nreset q[0];\n", 50)
    qpm.run(good, sync=False, on_done=on_done)
    qpm.run(bad, sync=False, on_done=on_done)

    assert finished.wait(10)
    assert sum(done[good].result().values()) == 50
    assert isinstance(done[bad].exception(), NotImplementedError)
    qpm.shutdown()


def test_cancel_and_shutdown_stop_queued_runs_only():
    qpm = QbloxTwinQPM(max_workers=1)
    gate = threading.Event()
    started = threading.Event()
    real_execute = qpm._execute

    def gated(cid):
        started.set()
        gate.wait(10)
        return real_execute(cid)

    qpm._execute = gated
    cancelled = []
    first = qpm.create_circuit(GHZ3, 10)
    second = qpm.create_circuit(GHZ3, 10)
    third = qpm.create_circuit(GHZ3, 10)
    qpm.run(first, sync=False)
    assert started.wait(10)
    qpm.run(second, sync=False, on_done=lambda c, f: cancelled.append(c))
    qpm.run(third, sync=False, on_done=lambda c, f: cancelled.append(c))

    assert qpm.read_queue() == [first, second, third]
    assert qpm.cancel(first) is False          # running: it finishes
    assert qpm.cancel(second) is True          # queued: it never runs
    assert cancelled == [second]               # reported on this thread
    assert qpm.cancel("never-queued") is False
    qpm.shutdown(wait=False)
    assert cancelled == [second, third]
    gate.set()
    assert sum(qpm.get_result(first, timeout=10).values()) == 10
