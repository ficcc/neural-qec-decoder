import json

import pytest

from src.runlog import compare, environment_info, record_run, summarise_results, summarise_training


def _write(p, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _row(dec, d, p, eps, scen="S2", inst=0, **extra):
    return {"decoder": dec, "d": d, "p": p, "scenario": scen, "instance": inst, "ler_round": eps, **extra}


def test_summarise_averages_seeds_and_instances(tmp_path):
    _write(tmp_path / "e2.jsonl", [_row("gnn_s0", 5, 0.004, 0.004, inst=0), _row("gnn_s1", 5, 0.004, 0.006, inst=0),
                                   _row("naive", 5, 0.004, 0.01, inst=0), _row("naive", 5, 0.004, 0.03, inst=1)])
    s = summarise_results(tmp_path)
    assert s["e2/S2/d5/p0.004/gnn"] == pytest.approx(0.005)
    assert s["e2/S2/d5/p0.004/naive"] == pytest.approx(0.02)


def test_summarise_keys_budget_and_skips_meta(tmp_path):
    _write(tmp_path / "e7.jsonl", [_row("finetuned_gnn", 5, 0.004, 0.006, scen="S3", budget=1000),
                                   _row("finetuned_gnn", 5, 0.004, 0.005, scen="S3", budget=10000),
                                   {"meta": "e1_summary", "threshold_d5_d7_naive": 0.0083}])
    s = summarise_results(tmp_path)
    assert s["e7/S3/d5/p0.004/n1000/finetuned_gnn"] == pytest.approx(0.006)
    assert s["e7/S3/d5/p0.004/n10000/finetuned_gnn"] == pytest.approx(0.005)
    assert s["e7/meta/threshold_d5_d7_naive"] == pytest.approx(0.0083)


def test_compare_reports_changes_and_missing_keys():
    a = {"x": 0.010, "y": 0.020, "only_a": 1.0}
    b = {"x": 0.011, "y": 0.020, "only_b": 2.0}
    rows = {r["key"]: r for r in compare(a, b)}
    assert rows["x"]["rel"] == pytest.approx(0.1)
    assert rows["y"]["rel"] == 0
    assert rows["only_a"]["b"] is None and rows["only_b"]["a"] is None
    assert [r["key"] for r in compare(a, b, min_rel=0.05)] == ["x", "only_a", "only_b"] or \
        {r["key"] for r in compare(a, b, min_rel=0.05)} == {"x", "only_a", "only_b"}


def test_summarise_training(tmp_path):
    d = tmp_path / "e1" / "d3_s0"; d.mkdir(parents=True)
    (d / "log.json").write_text(json.dumps({"best_val_ler": 0.057, "n_params": 133761, "wall_time": 270.0,
                                            "total_shots": 12288000, "history": [{"step": 12000}], "cfg": {"seed": 0}}))
    t = summarise_training(tmp_path)
    assert t["e1/d3_s0"] == {"best_val_ler": 0.057, "n_params": 133761, "wall_min": 4.5,
                             "steps": 12000, "total_shots": 12288000}


def test_environment_info_has_versions_and_git():
    e = environment_info()
    for k in ("python", "platform", "packages", "git", "timestamp_utc", "torch_cuda"):
        assert k in e
    assert {"stim", "pymatching", "torch", "numpy"} <= set(e["packages"])
    assert "commit" in e["git"] and "dirty" in e["git"]


def test_record_run_writes_self_contained_folder(tmp_path):
    res = tmp_path / "results"; res.mkdir()
    _write(res / "e1.jsonl", [_row("naive", 3, 0.002, 0.003, scen="S0")])
    out = record_run("t1", root=tmp_path / "runs", results_dir=res, ckpt_root=tmp_path / "none",
                     notes="n", extra={"cost_usd": 3.7}, test_result={"passed": 5, "failed": 0})
    m = json.loads((out / "manifest.json").read_text())
    assert m["label"] == "t1" and m["notes"] == "n" and m["extra"]["cost_usd"] == 3.7
    assert m["tests"] == {"passed": 5, "failed": 0} and "python" in m["environment"]
    assert json.loads((out / "summary.json").read_text())["metrics"]["e1/S0/d3/p0.002/naive"] == pytest.approx(0.003)
    assert (out / "results" / "e1.jsonl").exists()
