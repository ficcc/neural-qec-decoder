import json
from pathlib import Path

from scripts import make_paper_tables as mp


def test_sci_and_pm_formatting():
    assert mp.sci(3.014e-3) == "$3.01\\times10^{-3}$"
    assert mp.sci(0) == "$0$"
    assert mp.pm(5.49e-4, 3e-5) == r"$(5.49\,\pm\,0.30)\times10^{-4}$"


def test_generated_numbers_match_the_recorded_results(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "OUT", tmp_path)
    mp.main()
    macros = (tmp_path / "macros.tex").read_text()
    summary = json.loads(Path("results/e9_summary.json").read_text())
    summary_b = json.loads(Path("results/e9b_summary.json").read_text())
    assert f"\\newcommand{{\\nineRatio}}{{{summary['verdict']['ratio']:.2f}}}" in macros
    assert f"\\newcommand{{\\armBRatio}}{{{summary_b['verdict_extra']['ratio']:.3f}}}" in macros
    for name in ("e1", "e2", "e4", "e6", "e7", "e8", "e9", "e9b", "timing", "latency"):
        text = (tmp_path / "tables" / f"{name}.tex").read_text()
        assert text.startswith("\\begin{tabular}") and "nan" not in text.lower()
    assert "n/a" in (tmp_path / "tables" / "e6.tex").read_text()      # oracle correlated is omitted at d=3
