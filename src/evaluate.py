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
    errs = 0
    with torch.no_grad():
        for s in range(0, len(obs), batch):
            g = build_graph_batch(dets[s:s + batch], table, d=d, rounds=rounds, k=k)
            pred = (model(g) > 0).numpy()
            errs += int((pred != obs[s:s + batch]).sum())
    return errs


def evaluate_point(*, d: int, p: float, scenario: str, instance: int, test_circuit: stim.Circuit,
                   decoders: dict[str, Callable[[np.ndarray, np.ndarray], int]],
                   min_errors: int = 100, chunk: int = 100_000,
                   max_shots: int = 2_000_000) -> list[dict]:
    """Every decoder decodes the same chunks; stop once 'oracle' has min_errors (or max_shots)."""
    errors = dict.fromkeys(decoders, 0)
    shots, i = 0, 0
    while shots < max_shots:
        n = min(chunk, max_shots - shots)
        dets, obs = sample_shots(test_circuit, n, seed=test_seed(i))
        for name, fn in decoders.items():
            errors[name] += fn(dets, obs)
        shots += n; i += 1
        if "oracle" in errors and errors["oracle"] >= min_errors:
            break
    rows = []
    for name, e in errors.items():
        lo, hi = wilson(e, shots)
        rows.append({"decoder": name, "d": d, "p": p, "scenario": scenario, "instance": instance,
                     "shots": shots, "errors": e, "ler": e / shots, "lo": lo, "hi": hi,
                     "ler_round": per_round(e / shots, d)})
    return rows


def append_jsonl(rows: list[dict], path: Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
