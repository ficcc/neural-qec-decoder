"""Helpers shared by the evaluation scripts: sharding over points and merging shard outputs."""
import json
import os
from pathlib import Path

import torch

from experiments.jobs import ckpt_dir
from src.train import load_gnn


def my_points(points: list, shard: str) -> list:
    i, n = map(int, shard.split("/"))
    return [p for k, p in enumerate(points) if k % n == i]


def out_path(tag: str, shard: str, root: Path = Path("results")) -> Path:
    i, n = shard.split("/")
    p = Path(root) / "raw" / f"{tag}_shard{i}of{n}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_rows(tag: str, root: Path = Path("results")) -> list[dict]:
    rows = []
    for f in sorted((Path(root) / "raw").glob(f"{tag}_shard*.jsonl")):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return rows


def eval_device() -> str:
    """GNN inference device for evaluation: $NQD_EVAL_DEVICE, else CUDA when available."""
    return os.environ.get("NQD_EVAL_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def gnn(name: str):
    """(model, cfg) for a trained job, on the evaluation device; raises if the checkpoint is missing."""
    model, cfg = load_gnn(ckpt_dir(name) / "best.pt")
    return model.to(eval_device()), cfg
