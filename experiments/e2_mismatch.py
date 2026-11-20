"""E2: mismatched noise (S1 heterogeneous, S2 + bad couplers, S3 + drift).

  python -m experiments.e2_mismatch --shard i/n   # evaluate a share of the points
  python -m experiments.e2_mismatch --plot        # merge shards -> results/e2.jsonl, figures
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.common import gnn, load_rows, my_points, out_path
from experiments.jobs import E2_TEST_P, E2_TRAIN_GRID
from src.circuits import noiseless_circuit
from src.evaluate import append_jsonl, evaluate_point
from src.plots import base_name
from src.runner import gnn_decoder, matching_decoders
from src.scenarios import device_instance, load_scenarios
from src.train import TrainConfig, instance_seed, train_gnn, train_seed

SCEN, INST, DS = ("S1", "S2", "S3"), (0, 1, 2), (3, 5, 7)


def points():
    return [(s, i, d, p) for s in SCEN for i in INST for d in DS for p in E2_TEST_P]


def gnn_names(scen, inst, d):
    names = [f"e2/{scen}_i{inst}_d{d}_s0"]
    if scen == "S2" and inst == 0:
        names += [f"e2/S2_i0_d{d}_s1", f"e2/S2_i0_d{d}_s2"]
    return names


def run(pts, out, cal_shots, max_shots, models):
    scs = load_scenarios()
    for k, (scen, inst, d, p) in enumerate(pts):
        c = noiseless_circuit(d, d)
        t0, t1 = device_instance(c, p, scs[scen], instance_seed(inst, d))
        decs, test_c = matching_decoders(c, p, t0, t1, cal_shots=cal_shots,
                                         cal_seed=train_seed(91, 1000 * inst + 10 * d + k % 10))
        for name in models(scen, inst, d):
            model, cfg = gnn(name) if isinstance(name, str) else name
            decs[f"gnn_s{cfg.seed}"] = gnn_decoder(model, d, cfg.k)
        rows = evaluate_point(d=d, p=p, scenario=scen, instance=inst, test_circuit=test_c,
                              decoders=decs, max_shots=max_shots)
        append_jsonl(rows, out)
        print(f"{scen} i{inst} d={d} p={p}: " + ", ".join(f"{r['decoder']}={r['ler']:.2e}" for r in rows), flush=True)


def summarise(rows):
    """mean ± std over instances of per-round LER, per (scenario, d, p, decoder); GNN seeds averaged first."""
    per = defaultdict(list)
    for r in rows:
        per[(r["scenario"], r["d"], r["p"], base_name(r["decoder"]), r["instance"])].append(r["ler_round"])
    agg = defaultdict(list)
    for (s, d, p, dec, inst), v in per.items():
        agg[(s, d, p, dec)].append(np.mean(v))
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v)) for k, v in agg.items()}


def plot(rows):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summ = summarise(rows)
    decs = ["naive", "calibrated", "correlated", "gnn", "oracle", "oracle_corr"]
    fig, axes = plt.subplots(1, len(E2_TEST_P), figsize=(11, 4), sharey=True)
    for ax, p in zip(np.atleast_1d(axes), E2_TEST_P):
        x = np.arange(len(SCEN)); w = 0.14
        for j, dec in enumerate(decs):
            ratios, errs = [], []
            for s in SCEN:
                m, e, _ = summ.get((s, 5, p, dec), (np.nan, 0, 0))
                o = summ.get((s, 5, p, "oracle"), (np.nan, 0, 0))[0]
                ratios.append(m / o); errs.append(e / o)
            ax.bar(x + (j - 2.5) * w, ratios, w, yerr=errs, capsize=2, label=dec)
        ax.axhline(1, color="k", lw=0.8); ax.set_xticks(x, SCEN); ax.set_title(f"d=5, p={p}")
    np.atleast_1d(axes)[0].set_ylabel("LER / oracle-matching LER"); np.atleast_1d(axes)[0].legend(fontsize=7)
    Path("figures").mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig("figures/e2_gap_to_oracle.png", dpi=150); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    if a.quick:
        cfg = TrainConfig("S2", 0, (3,), E2_TRAIN_GRID, steps=300)
        ck = Path("checkpoints/e2_quick")
        if not (ck / "best.pt").exists():
            train_gnn(cfg, ck)
        from src.train import load_gnn
        m = load_gnn(ck / "best.pt")
        out = Path("results/raw/e2_quick.jsonl"); out.unlink(missing_ok=True)
        run([("S2", 0, 3, 0.004)], out, 20_000, 20_000, lambda *_: [m])
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        r = {x["decoder"]: x["ler"] for x in rows}
        assert r["oracle"] <= r["naive"] * 1.15, r
        print("quick ok")
        return
    if a.plot:
        rows = load_rows("e2")
        Path("results/e2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        summ = summarise(rows)
        for (s, d, p, dec), (m, e, n) in sorted(summ.items()):
            print(f"{s} d={d} p={p} {dec:11s} {m:.3e} ± {e:.1e} (n={n})")
        plot(rows)
        return
    run(my_points(points(), a.shard), out_path("e2", a.shard), 1_000_000, 2_000_000, gnn_names)


if __name__ == "__main__":
    main()
