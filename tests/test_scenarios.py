import numpy as np
import pytest

from src.circuits import noiseless_circuit
from src.noise import apply_noise, si1000_model
from src.scenarios import P_CAP, device_instance, load_scenarios

FIELDS = ("p_1q", "p_2q", "p_reset", "p_meas", "p_idle", "p_res_idle")
C = noiseless_circuit(5, 5)
SC = load_scenarios()


def _vals(m, f):
    return np.array(list(getattr(m, f).values()))


def test_s0_is_si1000():
    m, _ = device_instance(C, 0.004, SC["S0"], seed=3)
    ref = si1000_model(C, 0.004)
    for f in FIELDS + ("p_data",):
        assert np.allclose(_vals(m, f), _vals(ref, f))


def test_mean_preserved():
    m, _ = device_instance(C, 0.004, SC["S1"], seed=3)
    ref = si1000_model(C, 0.004)
    for f in FIELDS:
        assert _vals(m, f).mean() == pytest.approx(_vals(ref, f).mean())
        assert _vals(m, f).std() > 0
    assert (_vals(m, "p_data") == 0).all()


def test_same_seed_same_pattern_across_p():
    a, _ = device_instance(C, 0.002, SC["S2"], seed=7)
    b, _ = device_instance(C, 0.006, SC["S2"], seed=7)
    for f in FIELDS:
        assert np.allclose(_vals(a, f) / 0.002, _vals(b, f) / 0.006)


def test_bad_couplers_present():
    m, _ = device_instance(C, 0.004, SC["S2"], seed=7)
    v = _vals(m, "p_2q")
    assert v.max() / np.median(v) > 4


def test_drift_only_in_s3():
    t0, t1 = device_instance(C, 0.004, SC["S2"], seed=7)
    assert t0 == t1
    t0, t1 = device_instance(C, 0.004, SC["S3"], seed=7)
    assert t0 != t1
    ref = si1000_model(C, 0.004)
    for f in FIELDS:
        assert _vals(t1, f).mean() == pytest.approx(_vals(ref, f).mean())


def test_rates_clipped():
    t0, t1 = device_instance(C, 0.04, SC["S3"], seed=7)
    for m in (t0, t1):
        for f in FIELDS:
            assert _vals(m, f).max() <= P_CAP
        apply_noise(C, m).compile_detector_sampler().sample(5)


def test_different_seeds_differ():
    a, _ = device_instance(C, 0.004, SC["S1"], seed=1)
    b, _ = device_instance(C, 0.004, SC["S1"], seed=2)
    assert a != b


def test_s4_has_uniform_crosstalk_only():
    t0, t1 = device_instance(C, 0.003, SC["S4"], seed=1)
    ref = si1000_model(C, 0.003)
    assert t0 == t1
    for f in FIELDS:
        assert np.allclose(_vals(t0, f), _vals(ref, f))
    assert len(t0.p_xt) == 2 * 5 * 4 and all(v == pytest.approx(SC["S4"].xt * 0.003) for v in t0.p_xt.values())
    assert SC["S4"].xt == 0.25
