import json

from src.plots import plot_ler_vs_p


def _row(dec, d, p, eps):
    return {"decoder": dec, "d": d, "p": p, "ler": eps, "ler_round": eps, "lo": eps * 0.9, "hi": eps * 1.1}


def test_only_filter_limits_plotted_decoders(tmp_path):
    rows = [_row(dec, d, p, 0.01 * d / 3 + p) for dec in ("naive", "oracle", "gnn_s0", "gnn_s1", "correlated")
            for d in (3, 5) for p in (0.002, 0.004)]
    labels = plot_ler_vs_p(rows, tmp_path / "x.png", only=("naive", "gnn"))
    assert (tmp_path / "x.png").exists()
    assert sorted(labels) == ["gnn d=3", "gnn d=5", "naive d=3", "naive d=5"]
    assert "oracle d=3" in plot_ler_vs_p(rows, tmp_path / "y.png")
