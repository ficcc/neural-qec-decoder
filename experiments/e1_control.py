"""E1: uniform-noise control (S0). Threshold plot; GNN vs matching variants incl. correlated matching."""
import argparse
import json
from pathlib import Path

import numpy as np

from src.circuits import noiseless_circuit
from src.evaluate import append_jsonl, evaluate_point
from src.plots import aggregate, plot_ler_vs_p
from src.runner import gnn_decoder, matching_decoders
from src.scenarios import device_instance, load_scenarios
from src.train import TrainConfig, load_gnn, train_gnn, train_seed

P_GRID = (0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.010, 0.012)


def threshold(rows, da=5, db=7):
    agg = aggregate(rows)
    ps = sorted(k[2] for k in agg if k[0] == "naive" and k[1] == da)
    diff = [np.log(agg[("naive", db, p)][0]) - np.log(agg[("naive", da, p)][0]) for p in ps
            if ("naive", db, p) in agg and agg[("naive", db, p)][0] > 0 and agg[("naive", da, p)][0] > 0]
    for i in range(len(diff) - 1):
        if diff[i] < 0 <= diff[i + 1]:
            f = -diff[i] / (diff[i + 1] - diff[i])
            return float(np.exp(np.log(ps[i]) + f * (np.log(ps[i + 1]) - np.log(ps[i]))))
    return None


def get_gnn(cfg: TrainConfig, ckdir: Path):
    path = ckdir / "best.pt"
    if not path.exists():
        train_gnn(cfg, ckdir)
    return load_gnn(path)[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--d", type=int, nargs="*", default=[3, 5, 7])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--train-only", action="store_true")
    a = ap.parse_args(argv)
    ds, grid, seeds, steps, max_shots, cal = a.d, P_GRID, a.seeds, a.steps, 2_000_000, 1_000_000
    tag = "e1"
    if a.quick:
        ds, grid, seeds, steps, max_shots, cal, tag = [3], (0.006, 0.010), 1, 300, 20_000, 20_000, "e1_quick"
    sc = load_scenarios()["S0"]
    out = Path(f"results/{tag}.jsonl")
    if out.exists() and not a.train_only:
        out.unlink()
    models = {}
    for d in ds:
        for s in range(seeds):
            cfg = TrainConfig(scenario="S0", instance=0, d_list=(d,), p_grid=grid, steps=steps, seed=s)
            models[(d, s)] = get_gnn(cfg, Path(f"checkpoints/{tag}/d{d}_s{s}"))
    if a.train_only:
        return
    for d in ds:
        c = noiseless_circuit(d, d)
        for i, p in enumerate(grid):
            t0, t1 = device_instance(c, p, sc, 0)
            decs, test_c = matching_decoders(c, p, t0, t1, cal_shots=cal, cal_seed=train_seed(90, 100 * d + i))
            for s in range(seeds):
                decs[f"gnn_s{s}"] = gnn_decoder(models[(d, s)], d, k=8)
            rows = evaluate_point(d=d, p=p, scenario="S0", instance=0, test_circuit=test_c,
                                  decoders=decs, max_shots=max_shots)
            append_jsonl(rows, out)
            print(f"d={d} p={p}: " + ", ".join(f"{r['decoder']}={r['ler']:.2e}" for r in rows), flush=True)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    thr = threshold(rows) if {5, 7} <= set(ds) else None
    append_jsonl([{"meta": "threshold_d5_d7_naive", "value": thr, "correlated_matching": "available"}], out)
    print("threshold estimate:", thr)
    plot_ler_vs_p(rows, Path(f"figures/{tag}_threshold.png"), title="E1: uniform circuit noise (S0)")


if __name__ == "__main__":
    main()
