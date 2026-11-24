import json
import re
from pathlib import Path

import pytest

from scripts import make_eli5_data as md


def load(tmp_path, monkeypatch):
    monkeypatch.setattr(md, "OUT", tmp_path / "data.js")
    md.main()
    text = (tmp_path / "data.js").read_text(encoding="utf-8")
    assert text.startswith("window.RESULTS = ")
    return json.loads(re.sub(r"^window\.RESULTS = |;\s*$", "", text))


def test_structure_and_no_missing_values(tmp_path, monkeypatch):
    data = load(tmp_path, monkeypatch)
    assert set(data) >= {"e1", "e6", "e9"}
    e1 = data["e1"]
    assert e1["p"][0] == 0.0015 and 0.002 in e1["p"]
    for d in (3, 5, 7):
        cell = e1["ler"][f"{d},0.002"]
        assert set(cell) == {"naive", "calibrated", "correlated", "gnn"} and all(v > 0 for v in cell.values())
    assert "oracle_corr" not in json.dumps(data["e6"]["ler"]["3,0.0025"])      # omitted at d=3 (label conflicts)
    assert set(data["e9"]["sweep"]["0.002"]) == {str(t) for t in md.TAUS}


def test_numbers_match_the_recorded_summaries(tmp_path, monkeypatch):
    data = load(tmp_path, monkeypatch)
    b = json.loads(Path("results/e9b_summary.json").read_text())
    assert data["e9"]["arms"]["B"]["7,0.002"]["ratio"] == pytest.approx(b["summary_extra"]["7,0.002"]["ratio"])
    assert data["e9"]["arms"]["A"]["7,0.002"]["removed"] == pytest.approx(b["summary"]["7,0.002"]["removed"])
    sweep = data["e9"]["sweep"]["0.002"]["0.95"]
    assert 0 < sweep["removed"] < 1 and 0.9 < sweep["ratio"] < 1.1       # the exploratory sweep point quoted in the paper
