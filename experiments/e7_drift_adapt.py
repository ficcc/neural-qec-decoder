"""E7: adapting to drift with few post-drift shots (S3, d=5, p=0.004).

For n in {1e3, 1e4, 1e5} t1 shots: recalibrate matching (p_ij on the n shots) vs fine-tune the
stale S3 GNN (init from e2/S3_i*_d5_s0, lr 1e-4, <=2000 steps, fixed n-shot buffer; early stopping
uses a separate 2000-shot t1 validation set, reported as extra data). Baselines: stale calibrated,
stale GNN, oracle, oracle_corr (true t1 DEM)."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

from experiments.common import gnn, load_rows, my_points, out_path
from experiments.jobs import E2_TRAIN_GRID, ckpt_dir
from src.calibrate import calibrated_matching
from src.circuits import dem_of, noiseless_circuit
from src.evaluate import append_jsonl, count_errors_matching, evaluate_point
from src.noise import apply_noise, si1000_model
from src.runner import gnn_decoder, matching_decoders
from src.scenarios import device_instance, load_scenarios
from src.train import instance_seed, load_gnn, train_gnn, train_seed

D, P, BUDGETS, VAL = 5, 0.004, (1_000, 10_000, 100_000), 2_000


def points():
    return [(inst, n) for inst in (0, 1, 2) for n in BUDGETS]


def run(pts, out, max_shots, device):
    c = noiseless_circuit(D, D)
    sc = load_scenarios()["S3"]
    for inst, n in pts:
        t0, t1 = device_instance(c, P, sc, instance_seed(inst, D))
        decs, test_c = matching_decoders(c, P, t0, t1, cal_shots=1_000_000, cal_seed=train_seed(94, inst))
        decs = {k: decs[k] for k in ("naive", "oracle", "oracle_corr", "calibrated")}
        decs["stale_calibrated"] = decs.pop("calibrated")
        t1_dets = apply_noise(c, t1).compile_detector_sampler(seed=train_seed(95, 10 * inst + BUDGETS.index(n))).sample(n)
        recal = calibrated_matching(dem_of(apply_noise(c, si1000_model(c, P))), t1_dets)
        decs["recalibrated"] = lambda dets, obs, m=recal: count_errors_matching(m, dets, obs)
        stale_name = f"e2/S3_i{inst}_d{D}_s0"
        stale, cfg = gnn(stale_name)
        decs["stale_gnn"] = gnn_decoder(stale, D, cfg.k)
        ft_cfg = replace(cfg, use_t1=True, init_from=str(ckpt_dir(stale_name) / "best.pt"), lr=1e-4,
                         steps=2000, batch_size=256, val_shots=VAL, val_every=100, patience=5,
                         fixed_shots=n, device=device, p_grid=(P,), seed=100 + inst)
        ft_dir = ckpt_dir(f"e7/i{inst}_n{n}")
        if not (ft_dir / "best.pt").exists():
            train_gnn(ft_cfg, ft_dir)
        ft, _ = load_gnn(ft_dir / "best.pt")
        decs["finetuned_gnn"] = gnn_decoder(ft, D, cfg.k)
        rows = evaluate_point(d=D, p=P, scenario="S3", instance=inst, test_circuit=test_c,
                              decoders=decs, max_shots=max_shots)
        for r in rows:
            r |= {"budget": n, "val_shots_extra": VAL}
        append_jsonl(rows, out)
        print(f"i{inst} n={n}: " + ", ".join(f"{r['decoder']}={r['ler']:.2e}" for r in rows), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    if a.plot:
        rows = load_rows("e7")
        Path("results/e7.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        for r in sorted(rows, key=lambda r: (r["budget"], r["instance"], r["decoder"])):
            print(f"n={r['budget']:>6} i{r['instance']} {r['decoder']:16s} {r['ler_round']:.3e}")
        return
    run(my_points(points(), a.shard), out_path("e7", a.shard), 2_000_000, a.device)


if __name__ == "__main__":
    main()
