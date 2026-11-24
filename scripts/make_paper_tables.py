"""Generate the paper's LaTeX tables and headline-number macros from results/*.jsonl and results/*.json.

    python -m scripts.make_paper_tables            # writes docs/paper/tables/*.tex and docs/paper/macros.tex

Every number in these files comes from the recorded results; nothing is typed by hand.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.plots import base_name

OUT = Path("docs/paper")


def sci(x: float, digits: int = 2) -> str:
    if x == 0:
        return "$0$"
    exp = int(np.floor(np.log10(abs(x))))
    return f"${x / 10 ** exp:.{digits}f}\\times10^{{{exp}}}$"


def pm(mean: float, std: float, digits: int = 2) -> str:
    exp = int(np.floor(np.log10(abs(mean)))) if mean else 0
    m, s = f"{mean / 10 ** exp:.{digits}f}", f"{std / 10 ** exp:.{digits}f}"
    return "$(" + m + r"\,\pm\," + s + r")\times10^{" + str(exp) + "}$"


def load_rows(name: str) -> list[dict]:
    path = Path("results") / f"{name}.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if "decoder" in r]


def load_json(name: str):
    return json.loads((Path("results") / name).read_text(encoding="utf-8"))


def group(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[(base_name(r["decoder"]),) + key(r)].append(r["ler_round"])
    return g


def tabular(header: list[str], body: list[list[str]], align: str) -> str:
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in body]
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def table_e1(tables: dict, macros: dict) -> None:
    g = group(load_rows("e1"), key=lambda r: (r["d"], r["p"]))
    p = 0.002
    body = []
    for label, dec in (("naive MWPM", "naive"), ("calibrated MWPM", "calibrated"),
                       ("correlated MWPM", "correlated"), ("GNN (3 seeds)", "gnn")):
        cells = []
        for d in (3, 5, 7):
            v = g[(dec, d, p)]
            cells.append(pm(np.mean(v), np.std(v, ddof=1)) if len(v) > 1 else sci(v[0]))
        body.append([label] + cells)
    tables["e1"] = tabular(["Decoder", "$d=3$", "$d=5$", "$d=7$"], body, "lccc")
    meta = [json.loads(l) for l in Path("results/e1.jsonl").read_text(encoding="utf-8").splitlines() if '"meta"' in l][-1]
    macros["thrNaive"] = f"{100 * meta['threshold_d5_d7_naive']:.2f}"
    lam = meta["lambda"]["0.002"]
    macros["lamNaive"], macros["lamCorr"], macros["lamGnn"] = (f"{lam[k]:.1f}" for k in ("naive", "correlated", "gnn"))


def table_e2(tables: dict, macros: dict) -> None:
    rows = [r for r in load_rows("e2") if r["scenario"] == "S2" and r["p"] == 0.004]
    g = group(rows, key=lambda r: (r["d"], r["instance"]))
    body = []
    for label, dec in (("naive MWPM", "naive"), ("calibrated MWPM", "calibrated"), ("oracle MWPM", "oracle"),
                       ("correlated MWPM", "correlated"), ("oracle correlated MWPM", "oracle_corr"), ("GNN", "gnn")):
        cells = []
        for d in (3, 5, 7):
            per_inst = [np.mean(g[(dec, d, i)]) for i in range(3)]
            cells.append(sci(float(np.mean(per_inst))))
        body.append([label] + cells)
    tables["e2"] = tabular(["Decoder", "$d=3$", "$d=5$", "$d=7$"], body, "lccc")


def table_e4_e7(tables: dict, macros: dict) -> None:
    rows = load_rows("e4")
    g = defaultdict(list)
    for r in rows:
        g[(r["decoder"], r["budget"])].append(r["ler_round"])
    body = [[f"$10^{{{int(np.log10(b))}}}$", sci(np.mean(g[("calibrated", b)])), sci(np.mean(g[("gnn", b)]))]
            for b in (1000, 10000, 100000, 1000000)]
    tables["e4"] = tabular(["Shots", "Calibrated MWPM", "GNN"], body, "lcc")
    rows = load_rows("e7")
    g = defaultdict(list)
    for r in rows:
        g[(r["decoder"], r["budget"])].append(r["ler_round"])
    cols = ["stale_calibrated", "recalibrated", "stale_gnn", "finetuned_gnn", "oracle"]
    body = [[f"$10^{{{int(np.log10(b))}}}$"] + [sci(np.mean(g[(c, b)])) for c in cols] for b in (1000, 10000, 100000)]
    tables["e7"] = tabular(["New shots", "Stale cal.", "Recal.", "Stale GNN", "Fine-tuned GNN", "Oracle"], body, "lccccc")


def table_e6(tables: dict, macros: dict) -> None:
    g = group(load_rows("e6"), key=lambda r: (r["d"], r["p"]))
    p = 0.0025
    body = []
    for label, dec in (("naive MWPM", "naive"), ("calibrated MWPM", "calibrated"), ("oracle MWPM", "oracle"),
                       ("correlated MWPM", "correlated"), ("oracle correlated MWPM", "oracle_corr"), ("GNN", "gnn")):
        cells = []
        for d in (3, 5, 7):
            v = g.get((dec, d, p))
            cells.append(sci(float(np.mean(v))) if v else "n/a")
        body.append([label] + cells)
    tables["e6"] = tabular(["Decoder", "$d=3$", "$d=5$", "$d=7$"], body, "lccc")


def table_e8(tables: dict, macros: dict) -> None:
    g = group(load_rows("e8"), key=lambda r: (r["p"],))
    body = []
    for p in (0.002, 0.003, 0.004, 0.005, 0.006):
        def cell(dec):
            v = g[(dec, p)]
            return pm(np.mean(v), np.std(v, ddof=1)) if len(v) > 1 else sci(v[0])
        body.append([f"{p}", sci(float(np.mean(g[("naive", p)]))), cell("gnn_scratch"), cell("gnn_warm35")])
    tables["e8"] = tabular(["$p$", "Naive MWPM", "GNN from scratch", "GNN warm start"], body, "lccc")
    ratios = [np.mean(g[("gnn_warm35", p)]) / np.mean(g[("gnn_scratch", p)]) for p in (0.002, 0.003, 0.004, 0.005, 0.006)]
    macros["warmGain"] = f"{100 * (1 - np.mean(ratios)):.0f}"


def table_e9(tables: dict, macros: dict) -> None:
    s = load_json("e9_summary.json")
    body = []
    for p in (0.002, 0.003, 0.004, 0.005, 0.006):
        cells = [f"{p}"]
        for d in (3, 5, 7):
            v = s["summary"][f"{d},{p}"]
            cells.append(f"{v['ratio']:.3f} $\\pm$ {v['ratio_std']:.3f}")
        removed = " / ".join(f"{100 * s['summary'][f'{d},{p}']['removed']:.0f}\\%" for d in (3, 5, 7))
        body.append(cells + [removed])
    tables["e9"] = tabular(["$p$", "$d=3$", "$d=5$", "$d=7$ (zero-shot)", "Removed ($d=3/5/7$)"], body, "lcccc")
    v = s["verdict"]
    macros["nineRatio"] = f"{v['ratio']:.2f}"
    macros["nineRemoved"] = f"{100 * v['removed']:.0f}"
    rows = load_rows("e9")
    m = next(r for r in rows if r["d"] == 7 and r["p"] == 0.002 and r["decoder"] == "matching")
    macros["nineShots"], macros["nineErrors"] = f"{m['shots']:,}".replace(",", "\\,"), f"{m['errors']:,}".replace(",", "\\,")
    diag = load_json("e9_diag.json")
    pr = diag["precision_recall"]
    prec = [v["precision"] for v in pr.values()]
    rec = [v["recall"] for v in pr.values()]
    macros["precMin"], macros["precMax"] = f"{min(prec):.3f}", f"{max(prec):.3f}"
    macros["recMin"], macros["recMax"] = f"{min(rec):.2f}", f"{max(rec):.2f}"
    t = diag["timing_us_per_shot"]
    body = []
    for key, label in (("d5_p0.002", "$d=5,\\ p=0.002$"), ("d5_p0.004", "$d=5,\\ p=0.004$"),
                       ("d7_p0.002", "$d=7,\\ p=0.002$"), ("d7_p0.004", "$d=7,\\ p=0.004$")):
        x = t[key]
        body.append([label, f"{x['matching_full']:.1f}", f"{x['matching_residual']:.1f}",
                     f"{x['predecoder_inference']:.0f}", f"{x['removal']:.1f}", f"{x['predecoder_total']:.0f}"])
    tables["timing"] = tabular(["Setting", "MWPM full", "MWPM residual", "Pre-dec. inference", "Removal", "Pre-dec. total"],
                               body, "lccccc")
    ratio = [t[k]["predecoder_total"] / t[k]["matching_full"] for k in t]
    macros["slowMin"], macros["slowMax"] = f"{min(ratio):.0f}", f"{max(ratio):.0f}"


def table_e9b(tables: dict, macros: dict) -> None:
    s = load_json("e9b_summary.json")
    body = []
    for p in (0.002, 0.003, 0.004, 0.005, 0.006):
        a, b = s["summary"][f"7,{p}"], s["summary_extra"][f"7,{p}"]
        body.append([f"{p}", f"{a['ratio']:.3f} $\\pm$ {a['ratio_std']:.3f} ({100 * a['removed']:.0f}\\%)",
                     f"{b['ratio']:.3f} $\\pm$ {b['ratio_std']:.3f} ({100 * b['removed']:.0f}\\%)"])
    tables["e9b"] = tabular(["$p$", "Arm A, $\\tau=0.7$", "Arm B, $\\tau=0.95$"], body, "lcc")
    macros["armARatio"] = f"{s['verdict']['ratio']:.3f}"
    macros["armBRatio"] = f"{s['verdict_extra']['ratio']:.3f}"
    macros["armBRemoved"] = f"{100 * s['verdict_extra']['removed']:.0f}"
    rows = load_rows("e9b")
    m = next(r for r in rows if r["d"] == 7 and r["p"] == 0.002 and r["decoder"] == "matching")
    macros["nineBShots"], macros["nineBErrors"] = f"{m['shots']:,}".replace(",", "\\,"), f"{m['errors']:,}".replace(",", "\\,")


def table_latency(tables: dict, macros: dict) -> None:
    cpu, gpu = load_json("e5_cpu.json"), load_json("e5_cuda.json")
    body = []
    for rc, rg in zip(cpu["rows"], gpu["rows"]):
        body.append([f"{rc['d']}", f"{rc['mwpm_single']['median_us_round']:.2f}",
                     f"{rc['gnn_b1_total']['median_us_round']:.0f}", f"{rc['gnn_b64_total']['median_us_round']:.1f}",
                     f"{rc['gnn_b4096_total']['median_us_round']:.1f}", f"{rg['gnn_b4096_total']['median_us_round']:.1f}"])
    tables["latency"] = tabular(["$d$", "MWPM", "GNN $B{=}1$", "GNN $B{=}64$", "GNN $B{=}4096$", "GNN $B{=}4096$ (GPU)"],
                                body, "lccccc")


def main() -> None:
    tables, macros = {}, {}
    for fn in (table_e1, table_e2, table_e4_e7, table_e6, table_e8, table_e9, table_e9b, table_latency):
        fn(tables, macros)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    for name, text in tables.items():
        (OUT / "tables" / f"{name}.tex").write_text(text, encoding="utf-8")
    (OUT / "macros.tex").write_text(
        "% generated by scripts/make_paper_tables.py from results/; do not edit\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())), encoding="utf-8")
    print(f"wrote {len(tables)} tables and {len(macros)} macros to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
