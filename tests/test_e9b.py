import json

import pytest
import torch

import src.evaluate as ev
from experiments import e9_predecoder as e9
from src.circuits import noiseless_circuit
from src.noise import apply_noise, si1000_model
from src.predecoder import PreDecoder
from src.runlog import summarise_results
from src.seeds import SEEDS, test_seed
from src.train import val_seed


def test_choose_tau_per_p_requires_every_error_rate_to_pass():
    removed = {0.7: 0.80, 0.9: 0.60, 0.95: 0.45, 0.99: 0.15}
    ratio_by_p = {0.7: {0.002: 1.40, 0.004: 1.05, 0.006: 1.00},
                  0.9: {0.002: 1.04, 0.004: 0.97, 0.006: 0.96},     # fails at p=0.002 (> 1.02)
                  0.95: {0.002: 1.01, 0.004: 0.97, 0.006: 0.97},
                  0.99: {0.002: 1.00, 0.004: 0.99, 0.006: 0.99}}
    assert e9.choose_tau_per_p(removed, ratio_by_p, max_ratio=1.02) == 0.95
    assert e9.choose_tau_per_p(removed, {t: {0.002: 1.5} for t in removed}, max_ratio=1.02) == 0.99   # none -> largest


def test_select_tau_per_p_records_tables_on_fresh_validation_seeds(tmp_path, monkeypatch):
    seen = []
    real = e9.sample_shots
    monkeypatch.setattr(e9, "sample_shots", lambda c, n, seed: (seen.append(seed), real(c, n, seed))[1])
    torch.manual_seed(0)
    out = e9.select_tau([PreDecoder(hidden=8, layers=1)], d=3, val_shots=400, p_val=(0.004, 0.006),
                        out=tmp_path / "t.json", per_p=True, val_offset=e9.CONFIRM_VAL_OFFSET)
    assert seen and all(val_seed(e9.CONFIRM_VAL_OFFSET) <= s < test_seed(0) for s in seen)
    assert all(s >= val_seed(e9.CONFIRM_VAL_OFFSET) for s in seen)             # never the E9 selection seeds
    saved = json.loads((tmp_path / "t.json").read_text())
    assert set(saved["ratio_by_p"]) == {str(t) for t in e9.TAU_GRID}
    assert set(saved["ratio_by_p"]["0.7"]) == {"0.004", "0.006"}
    again = e9.choose_tau_per_p({float(t): v for t, v in saved["removed_frac"].items()},
                                {float(t): {float(p): r for p, r in d.items()} for t, d in saved["ratio_by_p"].items()})
    assert again == out["tau"] == saved["tau"] and saved["rule"] == "per_p"


def test_evaluate_point_chunk_offset_shifts_test_seeds(monkeypatch):
    seen = []
    real = ev.sample_shots
    monkeypatch.setattr(ev, "sample_shots", lambda c, n, seed: (seen.append(seed), real(c, n, seed))[1])
    c = noiseless_circuit(3, 3)
    noisy = apply_noise(c, si1000_model(c, 0.01))
    ev.evaluate_point(d=3, p=0.01, scenario="S0", instance=0, test_circuit=noisy,
                      decoders={"m": lambda dets, obs: 0}, min_errors=10 ** 9, chunk=200, max_shots=400,
                      chunk_offset=1000)
    assert seen == [test_seed(1000), test_seed(1001)]
    assert min(seen) > test_seed(999)


def test_confirmatory_points_refuse_without_committed_tau_file(tmp_path, monkeypatch):
    monkeypatch.setattr(e9, "TAU_FILE_B", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        e9.main(["--confirm-shard", "0/1"])


def test_confirmatory_run_uses_offset_and_has_no_sweep(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(e9, "evaluate_point", lambda **kw: calls.append(kw) or [])
    real_setup = e9.point_setup
    monkeypatch.setattr(e9, "point_setup", lambda d, p: real_setup(3, p))
    torch.manual_seed(0)
    models = [PreDecoder(hidden=8, layers=1)]
    e9.run_points([(7, 0.002)], tmp_path / "o.jsonl", 0.95, models, models[0], max_shots=999,
                  shots_dir=tmp_path, chunk_offset=e9.CONFIRM_CHUNK_OFFSET, sweep=False)
    assert calls[0]["chunk_offset"] == e9.CONFIRM_CHUNK_OFFSET >= 1000
    assert not any("_tau" in name for name in calls[0]["decoders"])
    assert {"matching", "pd_s0", "pd_ref"} <= set(calls[0]["decoders"])


def test_run_recorder_knows_the_confirmatory_experiment(tmp_path):
    row = {"decoder": "pd_s0", "d": 7, "p": 0.002, "scenario": "S0", "instance": 0, "ler_round": 0.001}
    (tmp_path / "e9b.jsonl").write_text(json.dumps(row) + "\n")
    assert summarise_results(tmp_path)["e9b/S0/d7/p0.002/pd"] == pytest.approx(0.001)
