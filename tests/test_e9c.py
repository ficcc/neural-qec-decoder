import torch

from experiments import e9_predecoder as e9
from src.predecoder import PreDecoder


def test_fixed_tau_constant_is_on_the_grid():
    assert e9.CONFIRM_FIXED_TAU == 0.95 and e9.CONFIRM_FIXED_TAU in e9.TAU_GRID


def test_run_points_adds_the_fixed_tau_arm(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(e9, "evaluate_point", lambda **kw: calls.append(kw) or [])
    real_setup = e9.point_setup
    monkeypatch.setattr(e9, "point_setup", lambda d, p: real_setup(3, p))
    torch.manual_seed(0)
    models = [PreDecoder(hidden=8, layers=1), PreDecoder(hidden=8, layers=1)]
    kw = dict(max_shots=999, shots_dir=tmp_path, sweep=False)
    e9.run_points([(7, 0.002)], tmp_path / "o.jsonl", 0.7, models, models[0], extra_taus=(0.95,), **kw)
    e9.run_points([(7, 0.002)], tmp_path / "o.jsonl", 0.7, models, models[0], **kw)
    with_arm, without = set(calls[0]["decoders"]), set(calls[1]["decoders"])
    assert {"matching", "pd_s0", "pd_s1", "pd_ref", "pdx0.95_s0", "pdx0.95_s1", "pdx0.95_ref"} <= with_arm
    assert not any(n.startswith("pdx") for n in without)


def _row(decoder, ler, removed=0.0, d=7, p=0.002):
    return {"decoder": decoder, "d": d, "p": p, "ler": ler, "removed_frac": removed, "scenario": "S0",
            "instance": 0, "ler_round": ler, "tau": None}


def test_summarise_prefix_selects_one_arm():
    rows = [_row("matching", 1e-3), _row("pd_s0", 0.9e-3, 0.8), _row("pdx0.95_s0", 1.1e-3, 0.5)]
    assert abs(e9.summarise(rows)[(7, 0.002)]["ratio"] - 0.9) < 1e-9
    assert abs(e9.summarise(rows, prefix="pdx0.95_s")[(7, 0.002)]["ratio"] - 1.1) < 1e-9
    assert abs(e9.summarise(rows, prefix="pdx0.95_s")[(7, 0.002)]["removed"] - 0.5) < 1e-9


def test_plot_results_reports_both_arms_with_their_own_verdicts(tmp_path):
    rows = [_row("matching", 1e-3), _row("pd_s0", 1.5e-3, 0.9), _row("pdx0.95_s0", 0.99e-3, 0.6)]
    out = e9.plot_results(rows, tmp_path, tmp_path / "figs", extra_prefix="pdx0.95_s")
    assert out["verdict"]["label"] == "partial"             # arm A: ratio 1.5 fails, removal passes
    assert out["verdict_extra"]["label"] == "positive"      # arm B: ratio 0.99 and 60% removed
