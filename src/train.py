"""On-the-fly GNN training from Stim samples, with early stopping on a fixed validation set."""
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import stim
import torch
import torch.nn.functional as F

from src.circuits import noiseless_circuit
from src.gnn import GNNDecoder
from src.graph import build_graph_batch, detector_table
from src.noise import apply_noise
from src.scenarios import device_instance, load_scenarios
from src.seeds import SEEDS


@dataclass(frozen=True)
class TrainConfig:
    scenario: str
    instance: int
    d_list: tuple[int, ...]
    p_grid: tuple[float, ...]
    steps: int = 5000
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
    instance_pool: tuple[int, ...] | None = None   # train across many random devices (E3)
    fixed_shots: int | None = None                 # cap on unique training shots (E4)


def instance_seed(instance: int, d: int) -> int:
    """Each code distance is its own device: a d=7 device is never a d=3 one."""
    return 1000 * instance + d


def train_seed(cfg_seed: int, idx: int) -> int:
    return SEEDS["train_base"] + 10_000 * cfg_seed + idx


def val_seed(idx: int) -> int:
    return SEEDS["val"] + idx


def make_circuits(cfg: TrainConfig) -> dict[tuple, stim.Circuit]:
    """Noisy t0 circuits keyed (d, p), or (d, p, device_seed) when training on an instance pool."""
    sc = load_scenarios()[cfg.scenario]
    out = {}
    for d in cfg.d_list:
        c = noiseless_circuit(d, d)
        for p in cfg.p_grid:
            if cfg.instance_pool is None:
                out[(d, p)] = apply_noise(c, device_instance(c, p, sc, instance_seed(cfg.instance, d))[0])
            else:
                for s in cfg.instance_pool:
                    out[(d, p, s)] = apply_noise(c, device_instance(c, p, sc, s)[0])
    return out


def _graph(dets, d, tables, k):
    return build_graph_batch(dets, tables[d], d=d, rounds=d, k=k)


def _evaluate(model, val, tables, k, device="cpu"):
    model.eval()
    losses, errs, n = 0.0, 0, 0
    with torch.no_grad():
        for d, dets, obs in val:
            logits = model(_graph(dets, d, tables, k).to(device)).cpu()
            y = torch.from_numpy(obs.astype(np.float32))
            losses += F.binary_cross_entropy_with_logits(logits, y, reduction="sum").item()
            errs += int(((logits > 0).numpy() != obs).sum())
            n += len(obs)
    model.train()
    return losses / n, errs / n


def train_gnn(cfg: TrainConfig, out_dir: Path) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    circuits = make_circuits(cfg)
    keys = list(circuits)
    tables = {d: detector_table(noiseless_circuit(d, d), d) for d in cfg.d_list}
    samplers = [circuits[key].compile_detector_sampler(seed=train_seed(cfg.seed, i))
                for i, key in enumerate(keys)]
    per = max(1, cfg.val_shots // len(keys))
    val = []
    for i, (d, p) in enumerate(keys):
        dets, obs = circuits[(d, p)].compile_detector_sampler(seed=val_seed(i)).sample(
            per, separate_observables=True)
        val.append((d, dets, obs[:, 0]))
    val_flip = float(np.mean(np.concatenate([o for _, _, o in val])))

    buffers = None
    if cfg.fixed_shots is not None:
        per_key = max(1, cfg.fixed_shots // len(keys))
        buffers = [s.sample(per_key, separate_observables=True) for s in samplers]
    unique = cfg.fixed_shots and per_key * len(keys)

    model = GNNDecoder(hidden=cfg.hidden, layers=cfg.layers).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best, bad, history = float("inf"), 0, []
    ckpt = out_dir / "best.pt"
    t0, total = time.perf_counter(), 0
    for step in range(1, cfg.steps + 1):
        i = int(rng.integers(len(keys)))
        d = keys[i][0]
        if buffers is None:
            dets, obs = samplers[i].sample(cfg.batch_size, separate_observables=True)
        else:
            idx = rng.integers(len(buffers[i][1]), size=cfg.batch_size)
            dets, obs = buffers[i][0][idx], buffers[i][1][idx]
        total += cfg.batch_size
        g = _graph(dets, d, tables, cfg.k).to(cfg.device)
        if not g.nonempty.any():
            continue
        logits = model(g)
        y = torch.from_numpy(obs[:, 0].astype(np.float32)).to(cfg.device)
        loss = F.binary_cross_entropy_with_logits(logits[g.nonempty], y[g.nonempty])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % cfg.val_every == 0 or step == cfg.steps:
            vl, ler = _evaluate(model, val, tables, cfg.k, cfg.device)
            history.append({"step": step, "train_loss": loss.item(), "val_bce": vl, "val_ler": ler})
            if vl < best:
                best, bad = vl, 0
                torch.save({"state_dict": model.state_dict(), "cfg": asdict(cfg), "val_ler": ler}, ckpt)
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    best_ler = min((h["val_ler"] for h in history if h["val_bce"] == best), default=float("nan"))
    (out_dir / "log.json").write_text(json.dumps({
        "cfg": asdict(cfg), "history": history, "n_params": model.n_params(),
        "total_shots": total, "unique_train_shots": unique if cfg.fixed_shots else total,
        "wall_time": time.perf_counter() - t0,
        "best_val_bce": best, "best_val_ler": best_ler, "val_flip_rate": val_flip}, indent=1))
    return ckpt


def load_gnn(path: Path) -> tuple[GNNDecoder, TrainConfig]:
    saved = torch.load(path, weights_only=False, map_location="cpu")
    raw = saved["cfg"]
    pool = raw.get("instance_pool")
    cfg = TrainConfig(**{**raw, "d_list": tuple(raw["d_list"]), "p_grid": tuple(raw["p_grid"]),
                         "instance_pool": tuple(pool) if pool is not None else None})
    model = GNNDecoder(hidden=cfg.hidden, layers=cfg.layers)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, cfg
