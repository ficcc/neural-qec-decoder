"""Matching decoders: from a DEM, and re-weighted from syndrome statistics (p_ij)."""
import numpy as np
import pymatching
import stim

P_MIN, P_MAX = 1e-6, 0.49


def matching_from_dem(dem: stim.DetectorErrorModel) -> pymatching.Matching:
    return pymatching.Matching.from_detector_error_model(dem)


def estimate_edge_probs(dets: np.ndarray, edges: list[tuple[int, int | None]]) -> np.ndarray:
    """Estimate each edge's flip probability from detection-event correlations.

    Bulk edge (i, j): p_ij = 1/2 - 1/2 sqrt(1 - 4(<x_i x_j> - <x_i><x_j>) /
                                         (1 - 2<x_i> - 2<x_j> + 4<x_i x_j>)).
    Boundary edge (i, None): the part of <x_i> not explained by i's bulk edges,
    using <x_i> = 1/2 - 1/2 prod_e (1 - 2 p_e).
    """
    x = dets.astype(np.float64)
    mean = x.mean(axis=0)
    second = (x.T @ x) / len(x)
    p = np.empty(len(edges))
    bulk_factor = np.ones(len(mean))   # prod over bulk edges at node of (1 - 2 p_e)
    for k, (i, j) in enumerate(edges):
        if j is None:
            continue
        a = second[i, j] - mean[i] * mean[j]
        b = 1 - 2 * mean[i] - 2 * mean[j] + 4 * second[i, j]
        if b <= 0:
            p[k] = P_MIN
        else:
            p[k] = 0.5 - 0.5 * np.sqrt(max(0.0, 1 - 4 * a / b))
        p[k] = np.clip(p[k], P_MIN, P_MAX)
        bulk_factor[i] *= 1 - 2 * p[k]
        bulk_factor[j] *= 1 - 2 * p[k]
    for k, (i, j) in enumerate(edges):
        if j is None:
            ratio = (1 - 2 * mean[i]) / bulk_factor[i]
            p[k] = np.clip(0.5 - 0.5 * ratio, P_MIN, P_MAX)
    return p


def calibrated_matching(naive_dem: stim.DetectorErrorModel, dets: np.ndarray) -> pymatching.Matching:
    """Naive DEM's graph structure and observables, with edge weights re-estimated from data."""
    edges = matching_from_dem(naive_dem).edges()
    probs = estimate_edge_probs(dets, [(u, v) for u, v, _ in edges])
    m = pymatching.Matching()
    for (u, v, attr), p in zip(edges, probs):
        w = float(np.log((1 - p) / p))
        if v is None:
            m.add_boundary_edge(u, fault_ids=attr["fault_ids"], weight=w, error_probability=float(p))
        else:
            m.add_edge(u, v, fault_ids=attr["fault_ids"], weight=w, error_probability=float(p))
    return m
