"""Regenerate the README's clean figures and print its numeric tables (markdown) from results/*.jsonl."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.plots import base_name, plot_ler_vs_p


def load(name):
    return [json.loads(l) for l in Path(f"results/{name}.jsonl").read_text().splitlines() if l.strip()]


def per_dec(rows, key):
    g = defaultdict(list)
    for r in rows:
        if "decoder" in r:
            g[(base_name(r["decoder"]),) + key(r)].append(r["ler_round"])
    return g


def fmt(v):
    return f"{np.mean(v):.2e}" + (f" ± {np.std(v, ddof=1):.1e}" if len(v) > 1 else "")


def main():
    e1, e6 = load("e1"), load("e6")
    plot_ler_vs_p(e1, Path("figures/e1_clean.png"), only=("naive", "correlated", "gnn"),
                  title="E1: SI1000 uniform noise — matching vs GNN")
    plot_ler_vs_p(e6, Path("figures/e6_clean.png"), only=("naive", "oracle", "oracle_corr", "gnn"),
                  title="E6: SI1000 + crosstalk")
    decs = ["naive", "calibrated", "correlated", "oracle_corr", "gnn"]
    for p in (0.002, 0.004):
        g = per_dec(e1, key=lambda r: (r["d"], r["p"]))
        print(f"\nE1 p={p}\n| decoder | d=3 | d=5 | d=7 |\n|---|---|---|---|")
        for dec in decs:
            print(f"| {dec} | " + " | ".join(fmt(g[(dec, d, p)]) for d in (3, 5, 7)) + " |")
    for scen in ("S1", "S2", "S3"):
        g = per_dec(load("e2"), key=lambda r: (r["scenario"], r["d"], r["p"], r["instance"]))
        print(f"\nE2 {scen}, p=0.004 (mean over 3 device instances)\n| decoder | d=3 | d=5 | d=7 |\n|---|---|---|---|")
        for dec in ["naive", "calibrated", "oracle", "correlated", "oracle_corr", "gnn"]:
            cells = []
            for d in (3, 5, 7):
                per_inst = [np.mean(g[(dec, scen, d, 0.004, i)]) for i in range(3)]
                cells.append(f"{np.mean(per_inst):.2e}")
            print(f"| {dec} | " + " | ".join(cells) + " |")
    g = per_dec(e6, key=lambda r: (r["d"], r["p"]))
    for p in (0.0015, 0.0025):
        print(f"\nE6 p={p}\n| decoder | d=3 | d=5 | d=7 |\n|---|---|---|---|")
        for dec in ["naive", "calibrated", "oracle", "correlated", "oracle_corr", "gnn"]:
            print(f"| {dec} | " + " | ".join(fmt(g[(dec, d, p)]) for d in (3, 5, 7)) + " |")
    e5 = {r: json.loads(Path(f"results/e5_{r}.json").read_text()) for r in ("cpu", "cuda")}
    print("\nE5 median µs/round (graph build + forward)\n| d | MWPM single | MWPM batch 4096 | GNN B=1 CPU | GNN B=64 CPU | GNN B=4096 CPU | GNN B=4096 GPU |\n|---|---|---|---|---|---|---|")
    for rc, rg in zip(e5["cpu"]["rows"], e5["cuda"]["rows"]):
        print(f"| {rc['d']} | {rc['mwpm_single']['median_us_round']:.2f} | {rc['mwpm_batch4096']['median_us_round']:.2f} | "
              f"{rc['gnn_b1_total']['median_us_round']:.0f} | {rc['gnn_b64_total']['median_us_round']:.1f} | "
              f"{rc['gnn_b4096_total']['median_us_round']:.1f} | {rg['gnn_b4096_total']['median_us_round']:.1f} |")
    print("\nE5 meta:", json.dumps(e5["cpu"]["meta"]), json.dumps(e5["cuda"]["meta"]))


if __name__ == "__main__":
    main()
