import json

import numpy as np
import pytest
import torch

from experiments import e9_predecoder as e9
from src.predecoder import PreDecoder
from src.seeds import SEEDS, test_seed


def test_choose_tau_picks_largest_removal_within_ratio():
    removed = {0.9: 0.80, 0.95: 0.70, 0.99: 0.55, 0.999: 0.30}
    ratio = {0.9: 1.20, 0.95: 1.04, 0.99: 1.01, 0.999: 1.00}
    assert e9.choose_tau(removed, ratio, max_ratio=1.02) == 0.99
    assert e9.choose_tau(removed, ratio, max_ratio=1.05) == 0.95
    assert e9.choose_tau(removed, {t: 1.5 for t in removed}, max_ratio=1.02) == 0.999   # none qualifies


def test_classify_result_all_branches():
    assert e9.classify_result(1.04, 0.35) == "positive"
    assert e9.classify_result(1.05, 0.30) == "positive"                # boundary values count
    assert e9.classify_result(1.04, 0.10) == "partial"
    assert e9.classify_result(1.20, 0.35) == "partial"
    assert e9.classify_result(1.20, 0.10) == "negative"


def test_test_points_refuse_without_tau_file(tmp_path, monkeypatch):
    monkeypatch.setattr(e9, "TAU_FILE", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        e9.main(["--shard", "0/1"])


def test_select_tau_uses_only_validation_seeds(tmp_path, monkeypatch):
    seen = []
    real = e9.sample_shots

    def spy(circuit, shots, seed):
        seen.append(seed)
        return real(circuit, shots, seed)

    monkeypatch.setattr(e9, "sample_shots", spy)
    torch.manual_seed(0)
    models = [PreDecoder(hidden=8, layers=1)]
    out = e9.select_tau(models, d=3, val_shots=600, p_val=(0.004,), out=tmp_path / "tau.json")
    assert seen and all(SEEDS["val"] <= s < test_seed(0) for s in seen)
    assert out["tau"] in e9.TAU_GRID and json.loads((tmp_path / "tau.json").read_text())["tau"] == out["tau"]
    assert set(out["removed_frac"]) == set(out["ratio"]) == {str(t) for t in e9.TAU_GRID}


@pytest.mark.slow
def test_quick_mode_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(e9, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(e9, "CKPT_QUICK", tmp_path / "ckpt")
    e9.main(["--quick"])
    rows = [json.loads(l) for l in (tmp_path / "results" / "raw" / "e9_quick.jsonl").read_text().splitlines()]
    names = {r["decoder"] for r in rows}
    assert "matching" in names and "pd_s0" in names
    assert (tmp_path / "results" / "e9_quick_tau.json").exists()
    pd = next(r for r in rows if r["decoder"] == "pd_s0")
    assert 0.0 <= pd["removed_frac"] <= 1.0 and pd["tau"] in e9.TAU_GRID
    assert (tmp_path / "results" / "shots_quick").exists()            # never the real shots directory
    assert not (tmp_path / "results" / "shots").exists()


def test_eval_budget_raises_power_at_the_verdict_point():
    b = e9.eval_budget(7, 0.002)
    assert b["min_errors"] == 1000 and b["max_shots"] >= 1_000_000
    assert e9.eval_budget(5, 0.002)["min_errors"] == 100 and e9.eval_budget(7, 0.004)["min_errors"] == 100


def test_budget_is_recorded_in_the_tau_file(tmp_path):
    torch.manual_seed(0)
    out = e9.select_tau([PreDecoder(hidden=8, layers=1)], d=3, val_shots=300, p_val=(0.004,), out=tmp_path / "tau.json")
    saved = json.loads((tmp_path / "tau.json").read_text())
    assert saved["eval_budget"]["7,0.002"]["min_errors"] == 1000 and out["eval_budget"] == saved["eval_budget"]


def test_run_points_passes_the_budget_to_evaluate_point(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(e9, "evaluate_point", lambda **kw: calls.append(kw) or [])
    real_setup = e9.point_setup
    monkeypatch.setattr(e9, "point_setup", lambda d, p: real_setup(3, p))      # keep the test small
    torch.manual_seed(0)
    models = [PreDecoder(hidden=8, layers=1)]
    for d, p in ((7, 0.002), (5, 0.004)):
        e9.run_points([(d, p)], tmp_path / "o.jsonl", 0.9, models, None, max_shots=999, shots_dir=tmp_path)
    assert calls[0]["min_errors"] == 1000 and calls[0]["max_shots"] >= 1_000_000
    assert calls[1]["min_errors"] == 100 and calls[1]["max_shots"] == 999


def test_time_components_includes_removal_cost():
    torch.manual_seed(0)
    c, noisy, table, det_table, matching = e9.point_setup(3, 0.004)
    dets, _ = e9.sample_shots(noisy, 500, seed=1)
    t = e9.time_components(PreDecoder(hidden=8, layers=1), table, det_table, matching, dets, 3, 0.5)
    for k in ("matching_full", "matching_residual", "predecoder_inference", "removal", "predecoder_total",
              "fired_per_shot", "residual_per_shot"):
        assert k in t and t[k] >= 0
    assert t["removal"] > 0 and t["predecoder_total"] == t["predecoder_inference"] + t["removal"]
