"""E4: data efficiency. Calibrated matching fitted on B shots vs GNN trained on B unique shots
(S2 instance 0, d=5, p=0.004), B in 1e3..1e6, 3 seeds each."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.common import gnn, load_rows, my_points, out_path
from experiments.jobs import E4_BUDGETS, E4_P
from src.circuits import noiseless_circuit
from src.evaluate import append_jsonl, evaluate_point
from src.runner import gnn_decoder, matching_decoders
from src.scenarios import device_instance, load_scenarios
from src.train import instance_seed, train_seed

D = 5


def points():
    return [(b, s) for b in E4_BUDGETS for s in range(3)]


def run(pts, out, max_shots):
    c = noiseless_circuit(D, D)
    t0, t1 = device_instance(c, E4_P, load_scenarios()["S2"], instance_seed(0, D))
    for b, s in pts:
        decs, test_c = matching_decoders(c, E4_P, t0, t1, cal_shots=b, cal_seed=train_seed(93, s))
        model, cfg = gnn(f"e4/B{b}_s{s}")
        decs = {"naive": decs["naive"], "oracle": decs["oracle"], "calibrated": decs["calibrated"],
                "gnn": gnn_decoder(model, D, cfg.k)}
        rows = evaluate_point(d=D, p=E4_P, scenario="S2", instance=0, test_circuit=test_c,
                              decoders=decs, max_shots=max_shots)
        for r in rows:
            r |= {"budget": b, "seed": s}
        append_jsonl(rows, out)
        print(f"B={b} s={s}: " + ", ".join(f"{r['decoder']}={r['ler']:.2e}" for r in rows), flush=True)


def plot(rows):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = defaultdict(list)
    for r in rows:
        g[(r["decoder"], r["budget"])].append(r["ler_round"])
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for dec, mk in (("calibrated", "s"), ("gnn", "D")):
        bs = sorted(b for d_, b in g if d_ == dec)
        ax.errorbar(bs, [np.mean(g[(dec, b)]) for b in bs], yerr=[np.std(g[(dec, b)], ddof=1) for b in bs],
                    marker=mk, capsize=3, label=dec)
    for dec, ls in (("oracle", "-"), ("naive", ":")):
        ax.axhline(np.mean([v for (d_, _), vs in g.items() if d_ == dec for v in vs]), color="k", ls=ls, lw=0.8, label=dec)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("shots used (calibration / unique training)")
    ax.set_ylabel("logical error rate per round"); ax.set_title(f"E4: S2, d={D}, p={E4_P}"); ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3); fig.tight_layout(); fig.savefig("figures/e4_budget.png", dpi=150); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args(argv)
    if a.plot:
        rows = load_rows("e4")
        Path("results/e4.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        plot(rows)
        return
    run(my_points(points(), a.shard), out_path("e4", a.shard), 2_000_000)


if __name__ == "__main__":
    main()
