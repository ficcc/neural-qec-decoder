"""Run recording: environment, code version, configs, training logs, test results and every key
metric in one folder (runs/<label>/), plus a differ so any two runs can be compared."""
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from src.plots import base_name

PACKAGES = ("stim", "pymatching", "sinter", "torch", "numpy", "matplotlib", "pytest")
EXPERIMENTS = ("e1", "e2", "e3", "e4", "e6", "e7", "e8", "e9", "e9b")


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def environment_info() -> dict:
    pk = {}
    for name in PACKAGES:
        try:
            pk[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pk[name] = None
    try:
        import torch
        cuda = {"available": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "threads": torch.get_num_threads()}
    except Exception:
        cuda = {"available": False, "device": None, "threads": None}
    return {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(), "packages": pk, "torch_cuda": cuda,
            "git": {"commit": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                    "dirty": bool(_git("status", "--porcelain"))},
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def summarise_results(results_dir: Path) -> dict[str, float]:
    """Mean per-round LER keyed exp/scenario/dD/pP[/nBUDGET]/decoder (seeds and instances averaged)."""
    groups = defaultdict(list)
    meta = {}
    for exp in EXPERIMENTS:
        f = Path(results_dir) / f"{exp}.jsonl"
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "decoder" not in r:
                for k, v in r.items():
                    if k != "meta" and isinstance(v, (int, float)):
                        meta[f"{exp}/meta/{k}"] = float(v)
                continue
            parts = [exp, r["scenario"], f"d{r['d']}", f"p{r['p']}"]
            if "budget" in r:
                parts.append(f"n{r['budget']}")
            parts.append(base_name(r["decoder"]))
            groups["/".join(parts)].append(r["ler_round"])
    return {**{k: float(np.mean(v)) for k, v in groups.items()}, **meta}


def summarise_training(ckpt_root: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(Path(ckpt_root).glob("*/*/log.json")):
        l = json.loads(f.read_text())
        name = f"{f.parent.parent.name}/{f.parent.name}"
        steps = l["history"][-1]["step"] if l.get("history") else None
        out[name] = {"best_val_ler": l.get("best_val_ler"), "n_params": l.get("n_params"),
                     "wall_min": round(l.get("wall_time", 0) / 60, 2), "steps": steps,
                     "total_shots": l.get("total_shots")}
    return out


def compare(a: dict, b: dict, min_rel: float = 0.0) -> list[dict]:
    """Per-key change from a to b, largest relative change first; keys in only one run have None."""
    rows = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k), b.get(k)
        rel = None if va in (None, 0) or vb is None else (vb - va) / va
        if rel is None or abs(rel) >= min_rel:
            rows.append({"key": k, "a": va, "b": vb, "rel": rel})
    return sorted(rows, key=lambda r: (r["rel"] is not None, -abs(r["rel"]) if r["rel"] is not None else 0))


def record_run(label: str, root: Path = Path("runs"), results_dir: Path = Path("results"),
               ckpt_root: Path = Path("checkpoints"), notes: str = "", extra: dict | None = None,
               test_result: dict | None = None) -> Path:
    out = Path(root) / label
    (out / "results").mkdir(parents=True, exist_ok=True)
    for f in Path(results_dir).glob("*.json*"):
        shutil.copy2(f, out / "results" / f.name)
    jobs = {}
    try:
        from dataclasses import asdict
        from experiments.jobs import all_jobs
        jobs = {n: {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(c).items()} for n, c in all_jobs()}
    except Exception:
        pass
    (out / "manifest.json").write_text(json.dumps({
        "label": label, "notes": notes, "extra": extra or {}, "environment": environment_info(),
        "tests": test_result, "job_configs": jobs}, indent=1))
    (out / "summary.json").write_text(json.dumps({
        "metrics": summarise_results(results_dir), "training": summarise_training(ckpt_root)}, indent=1))
    return out
