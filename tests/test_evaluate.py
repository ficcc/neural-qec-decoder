import hashlib

import numpy as np
import pytest

from src.calibrate import calibrated_matching, matching_from_dem
from src.circuits import dem_of, noiseless_circuit
from src.evaluate import count_errors_matching, evaluate_point, per_round, sample_shots, wilson
from src.noise import apply_noise, uniform_model
from src.scenarios import device_instance, load_scenarios
from src.seeds import SEEDS


def test_wilson_zero():
    lo, hi = wilson(0, 100)
    assert lo == 0.0 and hi == pytest.approx(0.0370, abs=1e-3)


def test_wilson_half():
    lo, hi = wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)


def test_per_round():
    assert per_round(0, 5) == 0
    assert per_round(0.1, 1) == pytest.approx(0.1)
    eps = per_round(0.2, 7)
    assert 0.5 * (1 - (1 - 2 * eps) ** 7) == pytest.approx(0.2)


def test_noiseless_zero_errors():
    c = noiseless_circuit(3, 3)
    noisy = apply_noise(c, uniform_model(c, 0.004))
    m = matching_from_dem(dem_of(noisy))
    dets, obs = sample_shots(c, 1000, seed=1)
    assert count_errors_matching(m, dets, obs) == 0


def test_all_decoders_same_shots():
    c = noiseless_circuit(3, 3)
    noisy = apply_noise(c, uniform_model(c, 0.01))
    seen = {"a": [], "oracle": []}

    def rec(name):
        def f(dets, obs):
            seen[name].append(hashlib.sha1(dets.tobytes()).hexdigest())
            return int(obs.sum())
        return f

    rows = evaluate_point(d=3, p=0.01, scenario="S0", instance=0, test_circuit=noisy,
                          decoders={"a": rec("a"), "oracle": rec("oracle")},
                          min_errors=10**9, chunk=500, max_shots=1500)
    assert seen["a"] == seen["oracle"] and len(seen["a"]) == 3
    assert len(set(seen["a"])) == 3
    r = {row["decoder"]: row for row in rows}
    assert r["a"]["shots"] == 1500
    assert {"decoder", "d", "p", "scenario", "instance", "shots", "errors", "ler", "lo", "hi",
            "ler_round"} <= set(r["a"])


@pytest.mark.slow
def test_calibrated_beats_naive_on_s2():
    d, p = 5, 0.006
    c = noiseless_circuit(d, d)
    t0, _ = device_instance(c, p, load_scenarios()["S2"], 1)
    true_c = apply_noise(c, t0)
    naive = matching_from_dem(dem_of(apply_noise(c, uniform_model(c, p))))
    cal_dets = true_c.compile_detector_sampler(seed=SEEDS["train_base"]).sample(200_000)
    cal = calibrated_matching(dem_of(apply_noise(c, uniform_model(c, p))), cal_dets)
    dets, obs = sample_shots(true_c, 200_000, seed=SEEDS["test"])
    n = len(obs)
    e_cal, e_naive = count_errors_matching(cal, dets, obs), count_errors_matching(naive, dets, obs)
    lo, hi = wilson(e_cal, n)
    assert e_cal / n <= e_naive / n + 2 * (hi - lo) / 2


def test_fit_lambda_recovers_synthetic():
    from src.evaluate import fit_lambda
    eps = {d: 0.03 * 2.0 ** (-(d + 1) / 2) for d in (3, 5, 7)}
    assert fit_lambda(eps) == pytest.approx(2.0)


def test_fit_lambda_ignores_zero_points():
    from src.evaluate import fit_lambda
    eps = {3: 0.01, 5: 0.005, 7: 0.0}
    assert fit_lambda(eps) == pytest.approx(2.0)
