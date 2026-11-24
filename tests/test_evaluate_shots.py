import numpy as np
import pytest

from src.circuits import noiseless_circuit
from src.evaluate import evaluate_point, load_errors, mcnemar_p, paired_compare
from src.noise import apply_noise, si1000_model


def test_evaluate_point_accepts_array_decoders_and_saves_shots(tmp_path):
    c = noiseless_circuit(3, 3)
    noisy = apply_noise(c, si1000_model(c, 0.01))

    def as_array(dets, obs):
        return dets[:, 0] != obs

    def as_count(dets, obs):
        return int((dets[:, 0] != obs).sum())

    out = tmp_path / "pt.npz"
    rows = evaluate_point(d=3, p=0.01, scenario="S0", instance=0, test_circuit=noisy,
                          decoders={"arr": as_array, "cnt": as_count}, min_errors=10**9,
                          chunk=500, max_shots=1500, save_errors=out)
    r = {x["decoder"]: x for x in rows}
    assert r["arr"]["errors"] == r["cnt"]["errors"] and r["arr"]["shots"] == 1500
    saved = load_errors(out)
    assert set(saved) == {"arr"}                      # only per-shot decoders are stored
    assert saved["arr"].shape == (1500,) and saved["arr"].dtype == bool
    assert int(saved["arr"].sum()) == r["arr"]["errors"]


def test_stop_on_names_the_reference_decoder():
    c = noiseless_circuit(3, 3)
    noisy = apply_noise(c, si1000_model(c, 0.02))
    rows = evaluate_point(d=3, p=0.02, scenario="S0", instance=0, test_circuit=noisy,
                          decoders={"matching": lambda dets, obs: dets[:, 0] != obs}, min_errors=20,
                          chunk=500, max_shots=5000, stop_on="matching")
    assert rows[0]["shots"] == 500                     # 20 errors reached within the first chunk


def test_mcnemar_known_values():
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(5, 5) == 1.0
    assert mcnemar_p(0, 10) == pytest.approx(2 * 0.5 ** 10)
    assert mcnemar_p(300, 200) < 1e-4
    assert mcnemar_p(3000, 2900) > 0.05                # normal-approximation branch (n > 1000)
    assert mcnemar_p(3000, 2500) < 1e-6


def test_paired_compare_identical_is_not_significant():
    e = np.random.default_rng(0).random(5000) < 0.05
    r = paired_compare(e, e.copy())
    assert r["only_a"] == r["only_b"] == 0 and r["diff"] == 0 and r["p_value"] == 1.0


def test_paired_compare_counts_discordant_pairs():
    a = np.array([1, 1, 0, 0, 1], dtype=bool)
    b = np.array([1, 0, 1, 0, 0], dtype=bool)
    r = paired_compare(a, b)
    assert (r["only_a"], r["only_b"]) == (2, 1) and r["diff"] == pytest.approx(1 / 5)
