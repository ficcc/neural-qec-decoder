"""Train every pending job from experiments.jobs (resumable: finished checkpoints are skipped).

Pod usage:  python -m experiments.train_jobs --min-d 5 --device cuda
Local d=3:  python -m experiments.train_jobs --max-d 3
"""
import argparse
import time
from dataclasses import replace

import torch

from experiments.jobs import all_jobs, ckpt_dir
from src.train import train_gnn


def select(jobs, min_d: int, max_d: int, only: tuple[str, ...]):
    return [(n, c) for n, c in jobs
            if min_d <= max(c.d_list) <= max_d and (not only or n.startswith(only))]


def pending(jobs):
    return [(n, c) for n, c in jobs
            if not ((ckpt_dir(n) / "best.pt").exists() and (ckpt_dir(n) / "log.json").exists())]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-d", type=int, default=0)
    ap.add_argument("--max-d", type=int, default=99)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=None, help="override steps (timing runs only)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    todo = pending(select(all_jobs(), a.min_d, a.max_d, tuple(a.only)))
    print(f"{len(todo)} pending jobs on {a.device}", flush=True)
    if a.list:
        for n, _ in todo:
            print(" ", n)
        return
    log = ckpt_dir("progress.log"); log.parent.mkdir(parents=True, exist_ok=True)
    for i, (name, cfg) in enumerate(todo, 1):
        cfg = replace(cfg, device=a.device, **({"steps": a.steps} if a.steps else {}))
        t0 = time.perf_counter()
        # Timing runs (--steps) go to a scratch dir so they can't masquerade as finished jobs.
        train_gnn(cfg, ckpt_dir(("_timing/" if a.steps else "") + name))
        line = f"[{i}/{len(todo)}] {name} done in {(time.perf_counter() - t0) / 60:.1f} min"
        print(line, flush=True)
        with log.open("a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
