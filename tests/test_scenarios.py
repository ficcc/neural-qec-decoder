import numpy as np
import pytest

from src.circuits import noiseless_circuit
from src.noise import apply_noise
from src.scenarios import P_CAP, device_instance, load_scenarios

FIELDS = ("p_1q", "p_2q", "p_reset", "p_meas", "p_data")
C = noiseless_circuit(5, 5)
SC = load_scenarios()


def _vals(m, f):
    return np.array(list(getattr(m, f).values()))


def test_s0_is_uniform():
    m, _ = device_instance(C, 0.004, SC["S0"], seed=3)
    for f in FIELDS:
        assert np.allclose(_vals(m, f), 0.004)


def test_mean_preserved():
    m, _ = device_instance(C, 0.004, SC["S1"], seed=3)
    for f in FIELDS:
        assert _vals(m, f).mean() == pytest.approx(0.004)
        assert _vals(m, f).std() > 0


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
    for f in FIELDS:
        assert _vals(t1, f).mean() == pytest.approx(0.004)


def test_rates_clipped():
    t0, t1 = device_instance(C, 0.1, SC["S3"], seed=7)
    for m in (t0, t1):
        for f in FIELDS:
            assert _vals(m, f).max() <= P_CAP
        apply_noise(C, m).compile_detector_sampler().sample(5)


def test_different_seeds_differ():
    a, _ = device_instance(C, 0.004, SC["S1"], seed=1)
    b, _ = device_instance(C, 0.004, SC["S1"], seed=2)
    assert a != b
