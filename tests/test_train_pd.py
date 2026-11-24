import json
import math

import pytest
import torch

from src.seeds import SEEDS, test_seed
from src.train import train_seed, val_seed
from src.train_pd import PDConfig, load_pd, train_pd


def _tiny(**kw):
    base = dict(scenario="S0", instance=0, d_list=(3,), p_grid=(0.004,), steps=6, batch_size=32,
                hidden=8, layers=1, val_shots=200, val_every=3)
    base.update(kw)
    return PDConfig(**base)


def test_defaults_follow_the_recipe():
    cfg = PDConfig("S0", 0, (3,), (0.004,))
    assert (cfg.steps, cfg.batch_size, cfg.hidden, cfg.layers, cfg.k, cfg.lr) == (12000, 1024, 64, 4, 8, 1e-3)
    assert cfg.pos_weight == 1.0     # positives are ~half of all candidates, so no re-weighting


def test_train_smoke_and_checkpoint_roundtrip(tmp_path):
    cfg = _tiny()
    path = train_pd(cfg, tmp_path)
    assert path.exists() and (tmp_path / "log.json").exists()
    model, cfg2 = load_pd(path)
    assert cfg2 == cfg
    saved = torch.load(path, weights_only=True)["state_dict"]
    for k, v in model.state_dict().items():
        assert torch.equal(v, saved[k])
    log = json.loads((tmp_path / "log.json").read_text())
    assert {"history", "n_params", "wall_time", "positive_rate", "best_val_bce"} <= set(log)
    assert {"step", "train_loss", "val_bce", "val_precision_at_0.9"} <= set(log["history"][0])


def test_validation_seeds_disjoint_from_training():
    assert train_seed(99, 49) < SEEDS["val"]
    assert val_seed(49) < test_seed(0)


def test_multi_distance_training_runs(tmp_path):
    path = train_pd(_tiny(d_list=(3, 5), steps=4, val_every=2), tmp_path)
    assert path.exists()


@pytest.mark.slow
def test_learns_something(tmp_path):
    cfg = _tiny(steps=400, batch_size=512, hidden=32, layers=2, val_shots=4000, val_every=100)
    train_pd(cfg, tmp_path)
    log = json.loads((tmp_path / "log.json").read_text())
    q = log["positive_rate"]
    prior_bce = -(q * math.log(q) + (1 - q) * math.log(1 - q))
    assert log["best_val_bce"] < 0.9 * prior_bce, (log["best_val_bce"], prior_bce)
