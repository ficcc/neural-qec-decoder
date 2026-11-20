import numpy as np
import pytest

from src.calibrate import calibrated_matching, matching_from_dem
from src.circuits import dem_of, noiseless_circuit
from src.noise import apply_noise, si1000_model
from src.scenarios import device_instance, load_scenarios
from src.seeds import SEEDS

SC = load_scenarios()


def _setup(d, p, scen, seed=1):
    c = noiseless_circuit(d, d)
    t0, _ = device_instance(c, p, SC[scen], seed)
    true_c = apply_noise(c, t0)
    naive_dem = dem_of(apply_noise(c, si1000_model(c, p)))
    return true_c, naive_dem


def _edge_probs(m):
    return {(u, v): a["error_probability"] for u, v, a in m.edges()}


def _sample(circ, n, seed):
    return circ.compile_detector_sampler(seed=seed).sample(n)


def _median_rel_err(est, ref, floor):
    keys = [k for k in ref if ref[k] > floor]
    return np.median([abs(est[k] - ref[k]) / ref[k] for k in keys])


@pytest.mark.slow
def test_recovers_true_edge_probs():
    true_c, naive_dem = _setup(3, 0.005, "S2")
    dets = _sample(true_c, 300_000, SEEDS["train_base"])
    est = _edge_probs(calibrated_matching(naive_dem, dets))
    ref = _edge_probs(matching_from_dem(dem_of(true_c)))
    assert set(est) == set(ref)
    assert _median_rel_err(est, ref, 2e-3) < 0.2


@pytest.mark.slow
def test_s0_close_to_naive():
    true_c, naive_dem = _setup(3, 0.005, "S0")
    dets = _sample(true_c, 300_000, SEEDS["train_base"])
    est = _edge_probs(calibrated_matching(naive_dem, dets))
    ref = _edge_probs(matching_from_dem(naive_dem))
    assert _median_rel_err(est, ref, 2e-3) < 0.15


def test_low_shot_finite():
    true_c, naive_dem = _setup(5, 0.004, "S2")
    m = calibrated_matching(naive_dem, _sample(true_c, 1000, SEEDS["train_base"]))
    for _, _, a in m.edges():
        assert np.isfinite(a["weight"])
        assert 1e-6 <= a["error_probability"] <= 0.49


def test_calibrated_keeps_observables():
    true_c, naive_dem = _setup(3, 0.004, "S0")
    m = calibrated_matching(naive_dem, _sample(true_c, 2000, 0))
    naive = {(u, v): a["fault_ids"] for u, v, a in matching_from_dem(naive_dem).edges()}
    assert {(u, v): a["fault_ids"] for u, v, a in m.edges()} == naive


def test_chunked_moments_match_unchunked():
    from src.calibrate import estimate_edge_probs
    true_c, naive_dem = _setup(3, 0.004, "S2")
    dets = _sample(true_c, 5000, 3)
    edges = [(u, v) for u, v, _ in matching_from_dem(naive_dem).edges()]
    a = estimate_edge_probs(dets, edges)
    b = estimate_edge_probs(dets, edges, chunk=700)
    assert np.allclose(a, b, rtol=1e-12, atol=1e-15)
