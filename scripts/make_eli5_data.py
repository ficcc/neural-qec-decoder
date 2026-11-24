"""Write docs/eli5/data.js: the real measured results used by the interactive explainer.

    python -m scripts.make_eli5_data

Every number comes from results/*.jsonl and results/e9b_summary.json; nothing is typed by hand.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.plots import base_name

OUT = Path("docs/eli5/data.js")
TAUS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)


def rows(name: str) -> list[dict]:
    path = Path("results") / f"{name}.jsonl"
    return [r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if "decoder" in r]


def mean_ler(rs: list[dict], decoders: tuple[str, ...]) -> dict:
    """{'d,p': {decoder: mean per-round LER}} (GNN seeds averaged)."""
    g = defaultdict(list)
    for r in rs:
        g[(r["d"], r["p"], base_name(r["decoder"]))].append(r["ler_round"])
    cells = {}
    for (d, p, dec), v in g.items():
        if dec in decoders:
            cells.setdefault(f"{d},{p}", {})[dec] = float(np.mean(v))
    return cells


def main() -> None:
    e1 = rows("e1")
    e6 = rows("e6")
    e9 = rows("e9")
    nine_b = json.loads(Path("results/e9b_summary.json").read_text(encoding="utf-8"))

    sweep = defaultdict(dict)
    matching = {r["p"]: r["ler"] for r in e9 if r["d"] == 7 and r["decoder"] == "matching"}
    for r in e9:
        if r["d"] == 7 and r["decoder"].startswith("pd_s0_tau"):
            sweep[str(r["p"])][str(r["tau"])] = {"removed": r["removed_frac"], "ratio": r["ler"] / matching[r["p"]]}

    arm = lambda s: {k: {"ratio": v["ratio"], "removed": v["removed"]} for k, v in s.items()}
    data = {
        "e1": {"p": sorted({r["p"] for r in e1}), "d": [3, 5, 7],
               "ler": mean_ler(e1, ("naive", "calibrated", "correlated", "gnn"))},
        "e6": {"p": sorted({r["p"] for r in e6}), "d": [3, 5, 7],
               "ler": mean_ler(e6, ("naive", "calibrated", "oracle", "correlated", "oracle_corr", "gnn"))},
        "e9": {"p": [0.002, 0.003, 0.004, 0.005, 0.006], "taus": list(TAUS), "sweep": dict(sweep),
               "arms": {"A": arm(nine_b["summary"]), "B": arm(nine_b["summary_extra"])}},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("window.RESULTS = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
