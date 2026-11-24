"""E8: does warm-starting d=7 from smaller-distance weights close the d=7 gap?

Training (GPU): python -m experiments.train_jobs --warmstart --device cuda
Evaluation:     python -m experiments.e8_warmstart --shard i/n ; python -m experiments.e8_warmstart --plot
Compared at d=7, S0, 12k steps each: GNN from scratch (e1/d7_s*), warm from d=3,5 (ws/d7_from_d35_s*),
warm from d=5 only (ws/d7_from_d5_s0), against matching baselines on identical shots.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.common import gnn, load_rows, my_points, out_path
from src.circuits import noiseless_circuit
from src.evaluate import append_jsonl, evaluate_point
from src.runner import gnn_decoder, matching_decoders
from src.scenarios import device_instance, load_scenarios
from src.train import train_seed

P_GRID, D = (0.002, 0.003, 0.004, 0.005, 0.006), 7
MODELS = {**{f"gnn_scratch_s{s}": f"e1/d7_s{s}" for s in range(3)},
          **{f"gnn_warm35_s{s}": f"ws/d7_from_d35_s{s}" for s in range(3)},
          "gnn_warm5_s0": "ws/d7_from_d5_s0"}


def steps_to_reach(history: list[dict], target: float):
    """First logged step whose validation LER is <= target, else None."""
    return next((h["step"] for h in history if h["val_ler"] <= target), None)


def points():
    return list(P_GRID)


def run(pts, out, max_shots=2_000_000):
    c = noiseless_circuit(D, D)
    sc = load_scenarios()["S0"]
    loaded = {label: gnn(name) for label, name in MODELS.items()}
    for k, p in enumerate(pts):
        t0, t1 = device_instance(c, p, sc, 0)
        # calibrated matching is reference only here, so a smaller calibration set is enough
        decs, test_c = matching_decoders(c, p, t0, t1, cal_shots=100_000, cal_seed=train_seed(96, k))
        for label, (model, cfg) in loaded.items():
            decs[label] = gnn_decoder(model, D, cfg.k)
        rows = evaluate_point(d=D, p=p, scenario="S0", instance=0, test_circuit=test_c,
                              decoders=decs, max_shots=max_shots)
        append_jsonl(rows, out)
        print(f"p={p}: " + ", ".join(f"{r['decoder']}={r['ler']:.2e}" for r in rows), flush=True)


def curves():
    from experiments.jobs import ckpt_dir
    out = {}
    for label, name in MODELS.items():
        f = ckpt_dir(name) / "log.json"
        if f.exists():
            out[label] = json.loads(f.read_text())["history"]
    return out


def plot_and_print():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cv = curves()
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for label, h in cv.items():
        kind = label.rsplit("_s", 1)[0]
        col = {"gnn_scratch": "tab:gray", "gnn_warm35": "tab:blue", "gnn_warm5": "tab:orange"}[kind]
        ax.plot([x["step"] for x in h], [x["val_ler"] for x in h], color=col, alpha=0.8,
                label=kind if label.endswith("s0") else None)
    ax.set_yscale("log"); ax.set_xlabel("d=7 training step"); ax.set_ylabel("validation LER per shot")
    ax.set_title("E8: d=7 training, scratch vs warm start"); ax.legend(); ax.grid(True, alpha=0.3)
    Path("figures").mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig("figures/e8_warmstart_curves.png", dpi=150); plt.close(fig)
    print("validation LER per shot (best checkpoint) and steps for the warm run to reach the scratch run's final value:")
    for s in range(3):
        sc = cv.get(f"gnn_scratch_s{s}"); w = cv.get(f"gnn_warm35_s{s}")
        if sc and w:
            tgt = min(x["val_ler"] for x in sc)
            print(f"  seed {s}: step0 {w[0]['val_ler']:.4f} | scratch best {tgt:.4f} | warm best "
                  f"{min(x['val_ler'] for x in w):.4f} | warm reaches scratch-best at step {steps_to_reach(w, tgt)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args(argv)
    if a.plot:
        rows = load_rows("e8")
        Path("results/e8.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        by = {}
        for r in rows:
            kind = r["decoder"].rsplit("_s", 1)[0] if r["decoder"].startswith("gnn") else r["decoder"]
            by.setdefault((r["p"], kind), []).append(r["ler_round"])
        for p in P_GRID:
            print(f"p={p}: " + ", ".join(f"{k}={np.mean(v):.3e}" for (pp, k), v in sorted(by.items()) if pp == p))
        plot_and_print()
        return
    run(my_points(points(), a.shard), out_path("e8", a.shard))


if __name__ == "__main__":
    main()
