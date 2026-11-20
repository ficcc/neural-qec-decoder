"""Plot helpers for LER curves."""
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLES = {"naive": ":", "oracle": "-", "calibrated": "--", "correlated": "-.", "gnn": "-"}
MARKERS = {"naive": "x", "oracle": "o", "calibrated": "s", "correlated": "^", "gnn": "D"}


def base_name(decoder: str) -> str:
    return re.sub(r"_s\d+$", "", decoder)


def aggregate(rows: list[dict], key=("d", "p"), metric: str = "ler_round") -> dict:
    """{(base_decoder, *key): (mean, err)}; err = std over seeds/instances if >1 row,
    else the Wilson half-width converted to the same metric."""
    groups = defaultdict(list)
    for r in rows:
        if "decoder" not in r:
            continue
        groups[(base_name(r["decoder"]),) + tuple(r[k] for k in key)].append(r)
    out = {}
    for g, rs in groups.items():
        vals = np.array([r[metric] for r in rs])
        if len(rs) > 1:
            out[g] = (vals.mean(), vals.std(ddof=1))
        else:
            r = rs[0]
            scale = r[metric] / r["ler"] if r["ler"] > 0 else 1.0
            out[g] = (vals[0], (r["hi"] - r["lo"]) / 2 * scale)
    return out


def plot_ler_vs_p(rows: list[dict], out: Path, per_round: bool = True, title: str = "") -> None:
    metric = "ler_round" if per_round else "ler"
    agg = aggregate(rows, metric=metric)
    ds = sorted({k[1] for k in agg}); decs = sorted({k[0] for k in agg})
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(ds)))
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for c, d in zip(colors, ds):
        for dec in decs:
            pts = sorted((k[2], v) for k, v in agg.items() if k[0] == dec and k[1] == d)
            if not pts:
                continue
            ps = [p for p, _ in pts]; m = [v[0] for _, v in pts]; e = [v[1] for _, v in pts]
            ax.errorbar(ps, m, yerr=e, color=c, ls=STYLES.get(dec, "-"), marker=MARKERS.get(dec, "."),
                        ms=4, capsize=2, label=f"{dec} d={d}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("physical error rate p")
    ax.set_ylabel("logical error rate per round" if per_round else "logical error rate per shot")
    ax.set_title(title); ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
