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
