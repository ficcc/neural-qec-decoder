"""On-the-fly training of the pre-decoder from fault-labelled Stim shots."""
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.circuits import dem_of, noiseless_circuit
from src.dem_labels import build_edge_table, fault_parity, isolated_faults, sample_with_faults
from src.graph import build_graph_batch, detector_table
from src.noise import apply_noise
from src.predecoder import PreDecoder, build_candidates, candidate_targets
from src.scenarios import device_instance, load_scenarios
from src.train import instance_seed, train_seed, val_seed


@dataclass(frozen=True)
class PDConfig:
    scenario: str
    instance: int
    d_list: tuple[int, ...]
    p_grid: tuple[float, ...]
    steps: int = 12000
    batch_size: int = 1024
    lr: float = 1e-3
    hidden: int = 64
    layers: int = 4
    k: int = 8
    seed: int = 0
    val_shots: int = 20000
    val_every: int = 500
    patience: int = 8
    device: str = "cpu"
    pos_weight: float = 1.0   # positives are ~half of all candidates, so no re-weighting by default


class _Bundle:
    """Everything needed to sample and label shots for one (d, p)."""

    def __init__(self, cfg: PDConfig, d: int, p: float):
        c = noiseless_circuit(d, d)
        t0, _ = device_instance(c, p, load_scenarios()[cfg.scenario], instance_seed(cfg.instance, d))
        self.dem = dem_of(apply_noise(c, t0))
        self.table = build_edge_table(self.dem)
        self.det_table = detector_table(c, d)
        self.d = d


def _batch_loss(model, g, c, iso, pos_weight):
    pair_logit, bnd_logit = model(g, c)
    pair_y, bnd_y = candidate_targets(g, c, iso)
    logits, y = torch.cat([pair_logit, bnd_logit]), torch.cat([pair_y, bnd_y])
    if len(y) == 0:
        return None, logits, y
    w = torch.where(y > 0.5, torch.full_like(y, pos_weight), torch.ones_like(y))
    return F.binary_cross_entropy_with_logits(logits, y, weight=w), logits, y


def _evaluate(model, val, cfg):
    """(mean BCE over candidates, precision among candidates with probability >= 0.9, positive rate)."""
    model.eval()
    tot, n, hi, hi_pos, pos = 0.0, 0, 0, 0.0, 0.0
    with torch.no_grad():
        for b, dets, iso in val:
            g = build_graph_batch(dets, b.det_table, d=b.d, rounds=b.d, k=cfg.k).to(cfg.device)
            c = build_candidates(g, b.table)
            _, logits, y = _batch_loss(model, g, c, iso, 1.0)
            if len(y) == 0:
                continue
            tot += F.binary_cross_entropy_with_logits(logits, y, reduction="sum").item()
            n += len(y)
            pos += y.sum().item()
            sel = torch.sigmoid(logits) >= 0.9
            hi += int(sel.sum())
            hi_pos += y[sel].sum().item()
    model.train()
    return tot / max(n, 1), (hi_pos / hi if hi else 0.0), pos / max(n, 1)


def train_pd(cfg: PDConfig, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    keys = [(d, p) for d in cfg.d_list for p in cfg.p_grid]
    bundles = [_Bundle(cfg, d, p) for d, p in keys]
    samplers = [b.dem.compile_sampler(seed=train_seed(cfg.seed, i)) for i, b in enumerate(bundles)]
    per = max(1, cfg.val_shots // len(keys))
    val = []
    for i, b in enumerate(bundles):
        dets, _, errs = sample_with_faults(b.dem, per, seed=val_seed(i))
        val.append((b, dets, isolated_faults(b.table, fault_parity(b.table, errs))))

    model = PreDecoder(hidden=cfg.hidden, layers=cfg.layers).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)
    best, bad, history, positive_rate = float("inf"), 0, [], float("nan")
    ckpt = out_dir / "best.pt"
    t0 = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        i = int(rng.integers(len(keys)))
        b = bundles[i]
        dets, _, errs = samplers[i].sample(cfg.batch_size, return_errors=True)
        iso = isolated_faults(b.table, fault_parity(b.table, errs))
        g = build_graph_batch(dets, b.det_table, d=b.d, rounds=b.d, k=cfg.k).to(cfg.device)
        c = build_candidates(g, b.table)
        loss, _, _ = _batch_loss(model, g, c, iso, cfg.pos_weight)
        if loss is None:
            continue
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % cfg.val_every == 0 or step == cfg.steps:
            vb, prec, positive_rate = _evaluate(model, val, cfg)
            history.append({"step": step, "train_loss": loss.item(), "val_bce": vb,
                            "val_precision_at_0.9": prec, "lr": sched.get_last_lr()[0]})
            if vb < best:
                best, bad = vb, 0
                torch.save({"state_dict": model.state_dict(), "cfg": asdict(cfg)}, ckpt)
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    (out_dir / "log.json").write_text(json.dumps({
        "cfg": asdict(cfg), "history": history, "n_params": model.n_params(),
        "wall_time": time.perf_counter() - t0, "best_val_bce": best, "positive_rate": positive_rate}, indent=1))
    return ckpt


def load_pd(path: Path) -> tuple[PreDecoder, PDConfig]:
    saved = torch.load(path, weights_only=True, map_location="cpu")
    raw = saved["cfg"]
    cfg = PDConfig(**{**raw, "d_list": tuple(raw["d_list"]), "p_grid": tuple(raw["p_grid"])})
    model = PreDecoder(hidden=cfg.hidden, layers=cfg.layers)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, cfg
