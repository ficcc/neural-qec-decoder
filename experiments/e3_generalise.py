"""E3: distance generalisation at d=7.

S0: GNN trained on d=3,5 (e3/S0_d35_s*) vs GNN trained at d=7 (e1/d7_s*), on the E1 grid subset.
S1: GNN trained across 20 random d=3,5 devices (e3/S1_d35_pool) and 20 d=7 devices (e3/S1_d7_pool),
    tested on 3 fresh, unseen d=7 devices against naive / calibrated / oracle matching.
"""
import argparse
import json
from pathlib import Path

from experiments.common import gnn, load_rows, my_points, out_path
from experiments.jobs import E2_TEST_P, E3_TEST_POOL
from src.circuits import noiseless_circuit
from src.evaluate import append_jsonl, evaluate_point
from src.plots import plot_ler_vs_p
from src.runner import gnn_decoder, matching_decoders
from src.scenarios import device_instance, load_scenarios
from src.train import train_seed

S0_P = (0.002, 0.003, 0.004, 0.005, 0.006)


def points():
    return ([("S0", 0, p) for p in S0_P] +
            [("S1", s, p) for s in E3_TEST_POOL for p in E2_TEST_P])


def run(pts, out, cal_shots, max_shots, d=7):
    scs = load_scenarios()
    c = noiseless_circuit(d, d)
    for k, (scen, dev_seed, p) in enumerate(pts):
        t0, t1 = device_instance(c, p, scs[scen], dev_seed)
        decs, test_c = matching_decoders(c, p, t0, t1, cal_shots=cal_shots, cal_seed=train_seed(92, k))
        if scen == "S0":
            names = {f"gnn35_s{s}": f"e3/S0_d35_s{s}" for s in range(3)}
            names |= {f"gnn7_s{s}": f"e1/d7_s{s}" for s in range(3)}
        else:
            names = {"gnn35pool": "e3/S1_d35_pool", "gnn7pool": "e3/S1_d7_pool"}
        for label, name in names.items():
            model, cfg = gnn(name)
            decs[label] = gnn_decoder(model, d, cfg.k)
        rows = evaluate_point(d=d, p=p, scenario=scen, instance=dev_seed, test_circuit=test_c,
                              decoders=decs, max_shots=max_shots)
        append_jsonl(rows, out)
        print(f"{scen} dev{dev_seed} p={p}: " + ", ".join(f"{r['decoder']}={r['ler']:.2e}" for r in rows), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args(argv)
    if a.plot:
        rows = load_rows("e3")
        Path("results/e3.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        plot_ler_vs_p([r for r in rows if r["scenario"] == "S0"], Path("figures/e3_generalise_S0.png"),
                      title="E3: d=7, GNN trained on d=3,5 vs on d=7 (S0)")
        for r in sorted(rows, key=lambda r: (r["scenario"], r["p"], r["instance"], r["decoder"])):
            if r["scenario"] == "S1":
                print(f"S1 dev{r['instance']} p={r['p']} {r['decoder']:10s} {r['ler_round']:.3e}")
        return
    run(my_points(points(), a.shard), out_path("e3", a.shard), 1_000_000, 2_000_000)


if __name__ == "__main__":
    main()
