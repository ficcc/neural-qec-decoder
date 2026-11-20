"""E6: S4 crosstalk (correlated XX/ZZ between neighbouring data qubits in different CX gates).
Matching must decompose these weight-2 faults into independent edges even with the true DEM;
oracle_corr (correlated matching on the true DEM) is the strongest classical baseline."""
import argparse
import json
from pathlib import Path

from experiments.common import load_rows, my_points, out_path
from experiments.e2_mismatch import run, summarise
from experiments.jobs import E6_TEST_P

DS = (3, 5, 7)


def points():
    return [("S4", 0, d, p) for d in DS for p in E6_TEST_P]


def models(scen, inst, d):
    return [f"e6/S4_d{d}_s{s}" for s in ((0, 1, 2) if d == 5 else (0,))]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args(argv)
    if a.plot:
        rows = load_rows("e6")
        Path("results/e6.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        for (s, d, p, dec), (m, e, n) in sorted(summarise(rows).items()):
            print(f"S4 d={d} p={p} {dec:11s} {m:.3e}")
        from src.plots import plot_ler_vs_p
        plot_ler_vs_p(rows, Path("figures/e6_crosstalk.png"), title="E6: SI1000 + crosstalk (S4)")
        return
    run(my_points(points(), a.shard), out_path("e6", a.shard), 1_000_000, 2_000_000, models)


if __name__ == "__main__":
    main()
