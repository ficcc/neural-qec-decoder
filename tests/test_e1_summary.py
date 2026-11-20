import pytest

from experiments.e1_control import lambdas, threshold


def _row(dec, d, p, eps):
    return {"decoder": dec, "d": d, "p": p, "ler": eps, "ler_round": eps, "lo": eps, "hi": eps}


def test_threshold_interpolates_crossing():
    rows = [_row("naive", 5, 0.004, 1e-3), _row("naive", 7, 0.004, 5e-4),
            _row("naive", 5, 0.016, 1e-2), _row("naive", 7, 0.016, 2e-2)]
    thr = threshold(rows)
    assert 0.004 < thr < 0.016


def test_lambdas_average_gnn_seeds():
    rows = [_row("gnn_s0", 3, 0.003, 4e-3), _row("gnn_s1", 3, 0.003, 4e-3),
            _row("gnn_s0", 5, 0.003, 2e-3), _row("gnn_s1", 5, 0.003, 2e-3)]
    assert lambdas(rows, 0.003)["gnn"] == pytest.approx(2.0)
