"""E1: uniform SI1000 control (S0). Threshold, Λ, and GNN vs naive/oracle/calibrated/correlated matching.

  python -m experiments.e1_control --shard i/n   # evaluate a share of the (d, p) points
  python -m experiments.e1_control --plot        # merge -> results/e1.jsonl, threshold, Λ, figure
  python -m experiments.e1_control --quick       # smoke test (d=3, short training)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.common import gnn, load_rows, my_points, out_path
from experiments.jobs import E1_GRID as P_GRID
from src.circuits import noiseless_circuit
from src.evaluate import append_jsonl, evaluate_point, fit_lambda
from src.plots import aggregate, base_name, plot_ler_vs_p
from src.runner import gnn_decoder, matching_decoders
from src.scenarios import device_instance, load_scenarios
from src.train import TrainConfig, load_gnn, train_gnn, train_seed

DS, SEEDS = (3, 5, 7), (0, 1, 2)


def threshold(rows, da=5, db=7):
    """p where the naive d=da and d=db per-round curves cross (log-log interpolation)."""
    agg = aggregate(rows)
    ps = sorted(k[2] for k in agg if k[0] == "naive" and k[1] == da)
    pts = [(p, np.log(agg[("naive", db, p)][0]) - np.log(agg[("naive", da, p)][0])) for p in ps
           if ("naive", db, p) in agg and agg[("naive", db, p)][0] > 0 and agg[("naive", da, p)][0] > 0]
    for (p0, d0), (p1, d1) in zip(pts, pts[1:]):
        if d0 < 0 <= d1:
            f = -d0 / (d1 - d0)
            return float(np.exp(np.log(p0) + f * (np.log(p1) - np.log(p0))))
    return None


def lambdas(rows, p):
    """Λ per decoder at physical error rate p (GNN seeds averaged per distance first)."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("p") == p:
            by[base_name(r["decoder"])][r["d"]].append(r["ler_round"])
    return {dec: fit_lambda({d: float(np.mean(v)) for d, v in ds.items()}) for dec, ds in by.items()}


def run(pts, out, cal_shots, max_shots, models):
    sc = load_scenarios()["S0"]
    for k, (d, p) in enumerate(pts):
        c = noiseless_circuit(d, d)
        t0, t1 = device_instance(c, p, sc, 0)
        decs, test_c = matching_decoders(c, p, t0, t1, cal_shots=cal_shots,
                                         cal_seed=train_seed(90, 100 * d + P_GRID.index(p) if p in P_GRID else k))
        for s, (model, cfg) in models(d).items():
            decs[f"gnn_s{s}"] = gnn_decoder(model, d, cfg.k)
        rows = evaluate_point(d=d, p=p, scenario="S0", instance=0, test_circuit=test_c,
                              decoders=decs, max_shots=max_shots)
        append_jsonl(rows, out)
        print(f"d={d} p={p}: " + ", ".join(f"{r['decoder']}={r['ler']:.2e}" for r in rows), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args(argv)
    if a.quick:
        ck = Path("checkpoints/e1_quick/d3_s0")
        if not (ck / "best.pt").exists():
            train_gnn(TrainConfig("S0", 0, (3,), (0.003, 0.006), steps=300), ck)
        m = load_gnn(ck / "best.pt")
        out = Path("results/raw/e1_quick.jsonl"); out.parent.mkdir(parents=True, exist_ok=True)
        out.unlink(missing_ok=True)
        run([(3, 0.003), (3, 0.006)], out, 20_000, 20_000, lambda d: {0: m})
        return
    if a.plot:
        rows = load_rows("e1")
        thr = threshold(rows)
        lam = {p: lambdas(rows, p) for p in (0.002, 0.003, 0.004)}
        meta = {"meta": "e1_summary", "threshold_d5_d7_naive": thr,
                "lambda": {str(p): v for p, v in lam.items()}}
        Path("results/e1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows + [meta]))
        print("threshold (naive d5/d7 crossing):", thr)
        for p, v in lam.items():
            print(f"Λ at p={p}: " + ", ".join(f"{k}={x:.2f}" for k, x in sorted(v.items())))
        plot_ler_vs_p(rows, Path("figures/e1_threshold.png"), title="E1: SI1000 uniform noise (S0)")
        return
    pts = [(d, p) for d in DS for p in P_GRID]
    run(my_points(pts, a.shard), out_path("e1", a.shard), 1_000_000, 2_000_000,
        lambda d: {s: gnn(f"e1/d{d}_s{s}") for s in SEEDS})


if __name__ == "__main__":
    main()
