"""E9: neural pre-decoder + matching.

  python -m experiments.e9_predecoder --select-tau        # choose tau on d=5 VALIDATION shots only
  python -m experiments.e9_predecoder --shard i/n         # test points (refuses to run without the tau file)
  python -m experiments.e9_predecoder --diag              # removal precision/recall and timing
  python -m experiments.e9_predecoder --plot              # merge shards, figures, paired tests, verdict
  python -m experiments.e9_predecoder --quick             # end-to-end smoke test (d=3, tiny model)

Models are trained with `python -m experiments.train_jobs --predecoder` (pd/d35_s0..2 on d=3,5 and
pd/d7_s0 as the d=7 reference). The residual syndrome is decoded with the same matching graph and
weights as the baseline (`robust_matching_from_dem` of the naive SI1000 DEM).
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.common import load_rows, my_points, out_path
from experiments.jobs import ckpt_dir
from src.circuits import dem_of, noiseless_circuit
from src.dem_labels import build_edge_table, fault_parity, isolated_faults, sample_with_faults
from src.dem_tools import robust_matching_from_dem
from src.evaluate import append_jsonl, evaluate_point, load_errors, paired_compare, sample_shots
from src.graph import detector_table
from src.noise import apply_noise, si1000_model
from src.predecoder import apply_tau, score
from src.seeds import test_seed
from src.train import val_seed
from src.train_pd import PDConfig, load_pd, train_pd

TAU_GRID = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)
P_VAL = (0.002, 0.004, 0.006)
P_TEST = (0.002, 0.003, 0.004, 0.005, 0.006)
DS = (3, 5, 7)
VAL_OFFSET = 500                       # validation shots use val_seed(500 + j): disjoint from training-time sets
RESULTS = Path("results")
TAU_FILE = RESULTS / "e9_tau.json"
CKPT_QUICK = Path("checkpoints/e9_quick")


def choose_tau(removed_frac: dict, ratio: dict, max_ratio: float = 1.02) -> float:
    """Largest removed fraction among taus whose LER ratio is within max_ratio; else the largest tau."""
    ok = [t for t in removed_frac if ratio[t] <= max_ratio]
    return max(ok, key=lambda t: removed_frac[t]) if ok else max(removed_frac)


def classify_result(ratio: float, removed_frac: float) -> str:
    """The pre-registered decision rule (spec section 5)."""
    good_ratio, good_removed = ratio <= 1.05, removed_frac >= 0.30
    return "positive" if good_ratio and good_removed else ("partial" if good_ratio or good_removed else "negative")


def point_setup(d: int, p: float):
    c = noiseless_circuit(d, d)
    noisy = apply_noise(c, si1000_model(c, p))      # S0: the true and the nominal model coincide
    dem = dem_of(noisy)
    return c, noisy, build_edge_table(dem), detector_table(c, d), robust_matching_from_dem(dem)


def _matching_errors(matching, dets, obs):
    return matching.decode_batch(dets)[:, 0].astype(bool) != obs


def select_tau(models, d: int = 5, val_shots: int = 100_000, p_val=P_VAL, out: Path | None = None) -> dict:
    """Pick tau on validation shots (paired with matching on the same shots), aggregated over models and p."""
    removed = {t: [0, 0] for t in TAU_GRID}
    errs_pd = dict.fromkeys(TAU_GRID, 0)
    errs_base = 0
    for j, p in enumerate(p_val):
        _, noisy, table, det_table, matching = point_setup(d, p)
        dets, obs = sample_shots(noisy, val_shots, seed=val_seed(VAL_OFFSET + j))
        base = int(_matching_errors(matching, dets, obs).sum())
        for model in models:
            sc = score(model, table, det_table, dets, d)
            errs_base += base
            for t in TAU_GRID:
                r = apply_tau(sc, table, dets, t)
                pred = matching.decode_batch(r.residual)[:, 0].astype(bool) ^ r.logical
                errs_pd[t] += int((pred != obs).sum())
                removed[t][0] += int(r.n_removed_dets.sum())
                removed[t][1] += int(r.n_fired.sum())
    removed_frac = {t: removed[t][0] / max(removed[t][1], 1) for t in TAU_GRID}
    ratio = {t: errs_pd[t] / max(errs_base, 1) for t in TAU_GRID}
    result = {"tau": choose_tau(removed_frac, ratio), "d": d, "p_val": list(p_val), "val_shots": val_shots,
              "n_models": len(models), "max_ratio": 1.02, "errors_matching": errs_base,
              "removed_frac": {str(t): v for t, v in removed_frac.items()},
              "ratio": {str(t): v for t, v in ratio.items()}}
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=1))
    return result


class _ScoreCache:
    """Network output for the current chunk, shared by all decoders using the same model."""

    def __init__(self):
        self.store = {}

    def get(self, key, model, dets, table, det_table, d):
        entry = self.store.get(key)
        if entry is not None and entry[0] is dets:      # holding the reference keeps ids unique
            return entry[1]
        sc = score(model, table, det_table, dets, d)
        self.store[key] = (dets, sc)
        return sc


def run_points(pts, out_jsonl: Path, tau: float, d35_models, ref_model, max_shots: int, shots_dir: Path):
    for d, p in pts:
        _, noisy, table, det_table, matching = point_setup(d, p)
        cache, stats, taus = _ScoreCache(), defaultdict(lambda: [0, 0]), {}
        decs = {"matching": lambda dets, obs: _matching_errors(matching, dets, obs)}

        def add_pd(name, key, model, t):
            taus[name] = t

            def fn(dets, obs):
                r = apply_tau(cache.get(key, model, dets, table, det_table, d), table, dets, t)
                stats[name][0] += int(r.n_removed_dets.sum())
                stats[name][1] += int(r.n_fired.sum())
                return (matching.decode_batch(r.residual)[:, 0].astype(bool) ^ r.logical) != obs
            decs[name] = fn

        for s, m in enumerate(d35_models):
            add_pd(f"pd_s{s}", ("d35", s), m, tau)
        if d == 7:
            if ref_model is not None:
                add_pd("pd_ref", ("ref", 0), ref_model, tau)
            for t in TAU_GRID:
                add_pd(f"pd_s0_tau{t}", ("d35", 0), d35_models[0], t)
        rows = evaluate_point(d=d, p=p, scenario="S0", instance=0, test_circuit=noisy, decoders=decs,
                              stop_on="matching", max_shots=max_shots,
                              save_errors=Path(shots_dir) / f"e9_d{d}_p{p}.npz")
        for r in rows:
            fired = stats[r["decoder"]][1]
            r["tau"] = taus.get(r["decoder"])
            r["removed_frac"] = stats[r["decoder"]][0] / fired if fired else 0.0
        append_jsonl(rows, out_jsonl)
        print(f"d={d} p={p}: " + ", ".join(f"{r['decoder']}={r['ler']:.2e}" for r in rows
                                           if "_tau" not in r["decoder"]), flush=True)


def summarise(rows: list[dict]) -> dict:
    """{(d, p): {ratio, ratio_std, removed, n}}: pre-decoder/matching LER ratio on identical shots,
    averaged over the d in {3,5}-trained seeds."""
    by = defaultdict(dict)
    for r in rows:
        if "decoder" in r:
            by[(r["d"], r["p"])][r["decoder"]] = r
    out = {}
    for key, dd in by.items():
        m = dd.get("matching")
        seeds = [r for n, r in dd.items() if n.startswith("pd_s") and "_tau" not in n]
        if not m or not m["ler"] or not seeds:
            continue
        ratios = [r["ler"] / m["ler"] for r in seeds]
        out[key] = {"ratio": float(np.mean(ratios)), "n": len(ratios),
                    "ratio_std": float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0,
                    "removed": float(np.mean([r["removed_frac"] for r in seeds]))}
    return out


def plot_results(rows: list[dict], shots_dir: Path, fig_dir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summ = summarise(rows)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for p in sorted({k[1] for k in summ}):
        ds = sorted(d for d, pp in summ if pp == p)
        ax.errorbar(ds, [summ[(d, p)]["ratio"] for d in ds], yerr=[summ[(d, p)]["ratio_std"] for d in ds],
                    marker="o", capsize=3, label=f"p={p}")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel("code distance d (d=7 is zero-shot)"); ax.set_ylabel("LER(pre-decoder + MWPM) / LER(MWPM)")
    ax.set_title("E9: pre-decoder trained on d=3,5"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "e9_ratio_vs_distance.png", dpi=150); plt.close(fig)

    sweep = defaultdict(dict)
    for r in rows:
        if "decoder" in r and r["d"] == 7 and r["decoder"].startswith("pd_s0_tau"):
            sweep[r["p"]][r["tau"]] = r
    base7 = {r["p"]: r for r in rows if "decoder" in r and r["d"] == 7 and r["decoder"] == "matching"}
    if sweep:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for p, byt in sorted(sweep.items()):
            ts = sorted(byt)
            ax.plot([byt[t]["removed_frac"] for t in ts], [byt[t]["ler"] / base7[p]["ler"] for t in ts],
                    marker="o", label=f"p={p}")
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xlabel("fraction of fired detectors removed"); ax.set_ylabel("LER ratio vs matching (d=7)")
        ax.set_title("E9: trade-off over the confidence threshold tau"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(fig_dir / "e9_tradeoff_d7.png", dpi=150); plt.close(fig)

    verdict = None
    if (7, 0.002) in summ:
        s = summ[(7, 0.002)]
        verdict = {"ratio": s["ratio"], "removed": s["removed"], "label": classify_result(s["ratio"], s["removed"])}
    paired = {}
    for p in sorted({k[1] for k in summ if k[0] == 7}):
        f = Path(shots_dir) / f"e9_d7_p{p}.npz"
        if f.exists():
            err = load_errors(f)
            paired[p] = {n: paired_compare(err[n], err["matching"]) for n in err if n.startswith("pd_s") and "_tau" not in n}
    return {"summary": {f"{d},{p}": v for (d, p), v in summ.items()}, "verdict": verdict,
            "paired_d7": {str(p): v for p, v in paired.items()}}


def diagnostics(models, tau: float, out: Path) -> dict:
    """Removal precision/recall of isolated faults (fault-labelled shots) and timing per shot."""
    res = {"tau": tau, "precision_recall": {}, "timing_us_per_shot": {}}
    for d in DS:
        for p in (0.002, 0.004):
            c = noiseless_circuit(d, d)
            dem = dem_of(apply_noise(c, si1000_model(c, p)))
            table, det_table = build_edge_table(dem), detector_table(c, d)
            dets, obs, errs = sample_with_faults(dem, 100_000, seed=test_seed(900 + d))
            iso = isolated_faults(table, fault_parity(table, errs))
            r = apply_tau(score(models[0], table, det_table, dets, d), table, dets, tau)
            hit = iso[r.removed_shot, r.removed_edge]
            res["precision_recall"][f"d{d}_p{p}"] = {
                "precision": float(hit.mean()) if len(hit) else None,
                "recall": float(hit.sum() / max(iso.sum(), 1)), "removed_frac": float(r.n_removed_dets.sum() / max(r.n_fired.sum(), 1))}
    for d in (5, 7):
        for p in (0.002, 0.004):
            _, noisy, table, det_table, matching = point_setup(d, p)
            dets, _ = sample_shots(noisy, 20_000, seed=test_seed(950 + d))

            def clock(fn, reps=3):
                ts = []
                for _ in range(reps):
                    t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
                return float(np.median(ts)) / len(dets) * 1e6
            sc = score(models[0], table, det_table, dets, d)
            r = apply_tau(sc, table, dets, tau)
            res["timing_us_per_shot"][f"d{d}_p{p}"] = {
                "matching_full": clock(lambda: matching.decode_batch(dets)),
                "matching_residual": clock(lambda: matching.decode_batch(r.residual)),
                "predecoder_inference": clock(lambda: score(models[0], table, det_table, dets, d), reps=1),
                "fired_per_shot": float(r.n_fired.mean()), "residual_per_shot": float((r.n_fired - r.n_removed_dets).mean())}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    return res


def _load_models():
    d35 = [load_pd(ckpt_dir(f"pd/d35_s{s}") / "best.pt")[0] for s in range(3)]
    ref = ckpt_dir("pd/d7_s0") / "best.pt"
    return d35, (load_pd(ref)[0] if ref.exists() else None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--select-tau", action="store_true")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    if a.quick:
        cfg = PDConfig("S0", 0, (3,), (0.004,), steps=300, batch_size=512, hidden=32, layers=2,
                       val_shots=3000, val_every=100)
        if not (CKPT_QUICK / "best.pt").exists():
            train_pd(cfg, CKPT_QUICK)
        model, _ = load_pd(CKPT_QUICK / "best.pt")
        tau_file = RESULTS / "e9_quick_tau.json"
        res = select_tau([model], d=3, val_shots=5000, p_val=(0.004,), out=tau_file)
        out = RESULTS / "raw" / "e9_quick.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.unlink(missing_ok=True)
        run_points([(3, 0.004)], out, res["tau"], [model], None, 20_000, RESULTS / "shots")
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        print(json.dumps(plot_results(rows, RESULTS / "shots", RESULTS / "figs")["summary"]))
        return
    if a.select_tau:
        d35, _ = _load_models()
        res = select_tau(d35, d=5, out=TAU_FILE)
        print("chosen tau:", res["tau"])
        return
    if a.plot:
        rows = load_rows("e9", root=RESULTS)
        (RESULTS / "e9.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        out = plot_results(rows, RESULTS / "shots", Path("figures"))
        (RESULTS / "e9_summary.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out["verdict"]))
        for p, v in out["paired_d7"].items():
            print(f"d=7 p={p}: " + "; ".join(f"{n}: diff {x['diff']:+.2e} (p={x['p_value']:.2g})" for n, x in v.items()))
        return
    if a.diag:
        d35, _ = _load_models()
        tau = json.loads(TAU_FILE.read_text())["tau"]
        diagnostics(d35, tau, RESULTS / "e9_diag.json")
        return
    if a.shard is not None:
        tau = json.loads(Path(TAU_FILE).read_text())["tau"]           # FileNotFoundError if not chosen yet
        d35, ref = _load_models()
        pts = my_points([(d, p) for d in DS for p in P_TEST], a.shard)
        run_points(pts, out_path("e9", a.shard, root=RESULTS), tau, d35, ref, 2_000_000, RESULTS / "shots")
        return
    ap.error("choose one of --select-tau, --shard, --diag, --plot, --quick")


if __name__ == "__main__":
    main()
