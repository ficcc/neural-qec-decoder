import json

import numpy as np
import pytest
import torch

from src.circuits import noiseless_circuit
from src.graph import build_graph_batch, detector_table
from src.noise import apply_noise
from src.scenarios import device_instance, load_scenarios
from src.seeds import SEEDS, test_seed
from src.train import TrainConfig, instance_seed, load_gnn, make_circuits, train_gnn, train_seed, val_seed


def test_seed_ranges_disjoint():
    assert train_seed(99, 49) < SEEDS["val"]
    assert val_seed(49) < test_seed(0)
    assert SEEDS["val"] <= val_seed(0)


def test_each_distance_own_device():
    cfg = TrainConfig(scenario="S2", instance=0, d_list=(3, 5), p_grid=(0.004,))
    circs = make_circuits(cfg)
    sc = load_scenarios()["S2"]
    for d in (3, 5):
        c = noiseless_circuit(d, d)
        t0, _ = device_instance(c, 0.004, sc, instance_seed(0, d))
        assert circs[(d, 0.004)] == apply_noise(c, t0)
    assert instance_seed(0, 3) != instance_seed(0, 5)


def test_checkpoint_roundtrip(tmp_path):
    cfg = TrainConfig(scenario="S0", instance=0, d_list=(3,), p_grid=(0.02,), steps=5,
                      batch_size=64, hidden=16, layers=2, val_shots=256, val_every=5)
    path = train_gnn(cfg, tmp_path)
    model, cfg2 = load_gnn(path)
    assert cfg2 == cfg
    t = detector_table(noiseless_circuit(3, 3), 3)
    dets = np.random.default_rng(0).random((32, len(t))) < 0.05
    g = build_graph_batch(dets, t, d=3, rounds=3, k=cfg.k)
    saved = torch.load(path, weights_only=False)
    ref = type(model)(hidden=16, layers=2)
    ref.load_state_dict(saved["state_dict"])
    model.eval(); ref.eval()
    assert torch.equal(model(g), ref(g))
    log = json.loads((tmp_path / "log.json").read_text())
    assert {"n_params", "total_shots", "wall_time", "best_val_ler", "val_flip_rate"} <= set(log)


@pytest.mark.slow
def test_learns_d3(tmp_path):
    cfg = TrainConfig(scenario="S0", instance=0, d_list=(3,), p_grid=(0.006,), steps=400,
                      batch_size=512, hidden=32, layers=3, val_shots=10_000, val_every=100)
    train_gnn(cfg, tmp_path)
    log = json.loads((tmp_path / "log.json").read_text())
    majority_err = min(log["val_flip_rate"], 1 - log["val_flip_rate"])
    assert log["best_val_ler"] < majority_err - 0.02


def test_device_field_roundtrips(tmp_path):
    cfg = TrainConfig(scenario="S0", instance=0, d_list=(3,), p_grid=(0.006,), steps=3,
                      batch_size=32, hidden=8, layers=1, val_shots=64, val_every=3, device="cpu")
    model, cfg2 = load_gnn(train_gnn(cfg, tmp_path))
    assert cfg2.device == "cpu" and next(model.parameters()).device.type == "cpu"


def test_instance_pool_samples_several_devices(tmp_path):
    cfg = TrainConfig(scenario="S1", instance=0, d_list=(3,), p_grid=(0.004,), instance_pool=(5000, 5001, 5002))
    circs = make_circuits(cfg)
    assert len(circs) == 3 and len(set(map(str, circs.values()))) == 3
    assert {k[2] for k in circs} == {5000, 5001, 5002}


def test_fixed_shot_budget_caps_shots(tmp_path):
    cfg = TrainConfig(scenario="S0", instance=0, d_list=(3,), p_grid=(0.006,), steps=50,
                      batch_size=64, hidden=8, layers=1, val_shots=64, val_every=10, fixed_shots=256)
    train_gnn(cfg, tmp_path)
    log = json.loads((tmp_path / "log.json").read_text())
    assert log["unique_train_shots"] == 256


def test_lr_decays_to_near_zero(tmp_path):
    cfg = TrainConfig(scenario="S0", instance=0, d_list=(3,), p_grid=(0.006,), steps=20,
                      batch_size=32, hidden=8, layers=1, val_shots=64, val_every=10)
    train_gnn(cfg, tmp_path)
    lrs = [h["lr"] for h in json.loads((tmp_path / "log.json").read_text())["history"]]
    assert lrs[0] < cfg.lr and lrs[-1] < 0.05 * cfg.lr


def test_default_budget_is_12k_steps():
    assert TrainConfig(scenario="S0", instance=0, d_list=(3,), p_grid=(0.004,)).steps == 12000
