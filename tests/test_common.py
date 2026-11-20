import json

from experiments.common import load_rows, my_points, out_path


def test_my_points_partition():
    pts = list(range(10))
    parts = [my_points(pts, f"{i}/3") for i in range(3)]
    assert sorted(x for p in parts for x in p) == pts


def test_load_rows_merges_shards(tmp_path):
    for i in range(2):
        out_path("e9", f"{i}/2", root=tmp_path).write_text(json.dumps({"decoder": "a", "k": i}) + "\n")
    rows = load_rows("e9", root=tmp_path)
    assert sorted(r["k"] for r in rows) == [0, 1]


def test_gnn_eval_device_follows_env(monkeypatch, tmp_path):
    import torch
    from experiments import common
    from src.train import TrainConfig, train_gnn
    monkeypatch.setattr(common, "ckpt_dir", lambda n: tmp_path / n)
    train_gnn(TrainConfig("S0", 0, (3,), (0.006,), steps=2, batch_size=16, hidden=8, layers=1,
                          val_shots=32, val_every=2), tmp_path / "x")
    monkeypatch.setenv("NQD_EVAL_DEVICE", "cpu")
    model, _ = common.gnn("x")
    assert next(model.parameters()).device == torch.device("cpu")
