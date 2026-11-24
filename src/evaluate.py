"""Shared-shot evaluation of all decoders, with Wilson intervals."""
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pymatching
import stim
import torch

from src.graph import build_graph_batch
from src.seeds import test_seed


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    ph = k / n
    den = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def per_round(p_shot: float, rounds: int) -> float:
    """Per-round logical error rate: eps = 1/2 (1 - (1 - 2 P_L)^(1/r))."""
    return 0.5 * (1 - max(0.0, 1 - 2 * p_shot) ** (1 / rounds))


def sample_shots(circuit: stim.Circuit, shots: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    dets, obs = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    return dets, obs[:, 0]


def count_errors_matching(m: pymatching.Matching, dets: np.ndarray, obs: np.ndarray) -> int:
    pred = m.decode_batch(dets)[:, 0]
    return int((pred != obs).sum())


def count_errors_gnn(model, dets: np.ndarray, obs: np.ndarray, table: np.ndarray, d: int,
                     rounds: int, k: int, batch: int = 4096) -> int:
    model.eval()
    device = next(model.parameters()).device
    if device.type == "cuda":
        batch = min(batch, 2048)   # many eval processes share one GPU
    errs = 0
    with torch.no_grad():
        for s in range(0, len(obs), batch):
            g = build_graph_batch(dets[s:s + batch], table, d=d, rounds=rounds, k=k).to(device)
            pred = (model(g) > 0).cpu().numpy()
            errs += int((pred != obs[s:s + batch]).sum())
    return errs


def evaluate_point(*, d: int, p: float, scenario: str, instance: int, test_circuit: stim.Circuit,
                   decoders: dict[str, Callable[[np.ndarray, np.ndarray], "int | np.ndarray"]],
                   min_errors: int = 100, chunk: int = 100_000, max_shots: int = 2_000_000,
                   stop_on: str = "oracle", save_errors: Path | None = None,
                   chunk_offset: int = 0) -> list[dict]:
    """Every decoder decodes the same chunks; stop once `stop_on` has min_errors (or max_shots).

    A decoder returns either an error count or a bool array of per-shot errors. Per-shot arrays are
    summed for the counts and, if `save_errors` is given, stored (packed bits) for paired comparisons.
    """
    errors = dict.fromkeys(decoders, 0)
    per_shot: dict[str, list[np.ndarray]] = {}
    shots, i = 0, 0
    while shots < max_shots:
        n = min(chunk, max_shots - shots)
        dets, obs = sample_shots(test_circuit, n, seed=test_seed(chunk_offset + i))
        for name, fn in decoders.items():
            res = fn(dets, obs)
            if isinstance(res, np.ndarray):
                per_shot.setdefault(name, []).append(res.astype(bool))
                errors[name] += int(res.sum())
            else:
                errors[name] += res
        shots += n; i += 1
        if stop_on in errors and errors[stop_on] >= min_errors:
            break
    if save_errors is not None and per_shot:
        Path(save_errors).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_errors, shots=shots,
                            **{k: np.packbits(np.concatenate(v)) for k, v in per_shot.items()})
    rows = []
    for name, e in errors.items():
        lo, hi = wilson(e, shots)
        rows.append({"decoder": name, "d": d, "p": p, "scenario": scenario, "instance": instance,
                     "shots": shots, "errors": e, "ler": e / shots, "lo": lo, "hi": hi,
                     "ler_round": per_round(e / shots, d)})
    return rows


def load_errors(path: Path) -> dict[str, np.ndarray]:
    """Per-shot bool error arrays saved by evaluate_point(save_errors=...)."""
    z = np.load(path)
    n = int(z["shots"])
    return {k: np.unpackbits(z[k])[:n].astype(bool) for k in z.files if k != "shots"}


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided p-value for discordant counts (b: only A wrong, c: only B wrong). Exact binomial
    up to b + c = 1000, continuity-corrected normal approximation beyond."""
    n = b + c
    if n == 0:
        return 1.0
    if n <= 1000:
        tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
        return min(1.0, 2 * tail)
    z = (abs(b - c) - 1) / math.sqrt(n)
    return math.erfc(z / math.sqrt(2))


def paired_compare(err_a: np.ndarray, err_b: np.ndarray) -> dict:
    """Paired comparison of two decoders on identical shots."""
    err_a, err_b = np.asarray(err_a, bool), np.asarray(err_b, bool)
    only_a, only_b = int((err_a & ~err_b).sum()), int((~err_a & err_b).sum())
    return {"only_a": only_a, "only_b": only_b, "diff": (only_a - only_b) / len(err_a),
            "p_value": mcnemar_p(only_a, only_b)}


def append_jsonl(rows: list[dict], path: Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def fit_lambda(eps_by_d: dict[int, float]) -> float:
    """Error-suppression factor from eps_L(d) ∝ Λ^{-(d+1)/2}: log-linear least squares.
    Points with eps <= 0 (no observed errors) are excluded."""
    pts = [(d, e) for d, e in sorted(eps_by_d.items()) if e > 0]
    if len(pts) < 2:
        return float("nan")
    x = np.array([(d + 1) / 2 for d, _ in pts])
    y = np.log([e for _, e in pts])
    slope = np.polyfit(x, y, 1)[0]
    return float(np.exp(-slope))
