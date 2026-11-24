"""Compare two recorded runs:  python -m scripts.compare_runs runs/A runs/B [--min-rel 0.05] [--filter e6/]"""
import argparse
import json
from pathlib import Path

from src.runlog import compare


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--min-rel", type=float, default=0.0, help="hide changes smaller than this (0.05 = 5%%)")
    ap.add_argument("--filter", default="", help="only keys containing this text")
    ap.add_argument("--training", action="store_true", help="compare training best_val_ler instead of test LER")
    a = ap.parse_args(argv)
    sa, sb = (json.loads((Path(r) / "summary.json").read_text()) for r in (a.a, a.b))
    ma, mb = (json.loads((Path(r) / "manifest.json").read_text()) for r in (a.a, a.b))
    print(f"A: {ma['label']}  commit {ma['environment']['git']['commit'][:8]}  {ma['environment']['timestamp_utc']}")
    print(f"B: {mb['label']}  commit {mb['environment']['git']['commit'][:8]}  {mb['environment']['timestamp_utc']}")
    for pkg in sorted(set(ma["environment"]["packages"]) | set(mb["environment"]["packages"])):
        va, vb = ma["environment"]["packages"].get(pkg), mb["environment"]["packages"].get(pkg)
        if va != vb:
            print(f"  env differs: {pkg} {va} -> {vb}")
    if a.training:
        ka = {k: v["best_val_ler"] for k, v in sa["training"].items()}
        kb = {k: v["best_val_ler"] for k, v in sb["training"].items()}
    else:
        ka, kb = sa["metrics"], sb["metrics"]
    rows = [r for r in compare(ka, kb, a.min_rel) if a.filter in r["key"]]
    print(f"{len(rows)} differing metrics")
    for r in rows:
        rel = "   new/removed" if r["rel"] is None else f"{r['rel']:+8.1%}"
        fa = "-" if r["a"] is None else f"{r['a']:.3e}"; fb = "-" if r["b"] is None else f"{r['b']:.3e}"
        print(f"  {r['key']:55s} {fa:>10} -> {fb:>10} {rel}")


if __name__ == "__main__":
    main()
