"""Fault-level labels: which matching-graph edges fired in a shot, and which faults are isolated.

Stim's DEM sampler can return, per shot, which error mechanisms fired. Each mechanism contributes
its graph-like components (matching-graph edges). A fault is *isolated* when its edge fired and no
other fired edge touches either of its detectors, so it is the unambiguous explanation of them.
"""
from dataclasses import dataclass

import numpy as np
import stim

from src.dem_tools import _group, mechanism_components


@dataclass
class EdgeTable:
    n_det: int
    u: np.ndarray          # (E,) int64, smaller detector id
    v: np.ndarray          # (E,) int64, larger detector id, -1 for a boundary edge
    label: np.ndarray      # (E,) uint8, logical-flip bit (observable 0) of the most probable mechanism
    edge_id: np.ndarray    # (n_det, n_det) int32, symmetric, -1 where there is no pair edge
    bnd_id: np.ndarray     # (n_det,) int32, -1 where there is no boundary edge
    mech_ptr: np.ndarray   # (M + 1,) int64, CSR over DEM mechanisms
    mech_edge: np.ndarray  # (nnz,) int32, edge id of each mechanism component


def build_edge_table(dem: stim.DetectorErrorModel) -> EdgeTable:
    groups = _group(dem)
    keys = sorted(groups, key=lambda k: (k[0], -1 if k[1] is None else k[1]))
    index = {k: i for i, k in enumerate(keys)}
    n_det = dem.num_detectors
    u = np.array([k[0] for k in keys], dtype=np.int64)
    v = np.array([-1 if k[1] is None else k[1] for k in keys], dtype=np.int64)
    label = np.array([1 if 0 in max(groups[k], key=lambda x: x[0])[1] else 0 for k in keys], dtype=np.uint8)
    edge_id = -np.ones((n_det, n_det), dtype=np.int32)
    bnd_id = -np.ones(n_det, dtype=np.int32)
    for i, (a, b) in enumerate(keys):
        if b is None:
            bnd_id[a] = i
        else:
            edge_id[a, b] = edge_id[b, a] = i
    mech = mechanism_components(dem)
    mech_ptr = np.zeros(len(mech) + 1, dtype=np.int64)
    mech_ptr[1:] = np.cumsum([len(c) for c in mech])
    mech_edge = np.array([index[c] for comps in mech for c in comps], dtype=np.int32)
    return EdgeTable(n_det, u, v, label, edge_id, bnd_id, mech_ptr, mech_edge)


def fault_parity(table: EdgeTable, errs: np.ndarray) -> np.ndarray:
    """(B, E) uint8: parity of fired components per edge, from the sparse list of fired mechanisms."""
    B, E = errs.shape[0], len(table.u)
    shot, mech = np.nonzero(errs)
    counts = table.mech_ptr[mech + 1] - table.mech_ptr[mech]
    total = int(counts.sum())
    par = np.zeros((B, E), dtype=np.uint8)
    if total:
        within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        edges = table.mech_edge[np.repeat(table.mech_ptr[mech], counts) + within]
        np.bitwise_xor.at(par, (np.repeat(shot, counts), edges), 1)
    return par


def isolated_faults(table: EdgeTable, parity: np.ndarray) -> np.ndarray:
    """(B, E) bool: edge fired (parity 1) and each of its detectors is touched by exactly one fired edge."""
    B, E = parity.shape
    shot, e = np.nonzero(parity.astype(bool))
    touch = np.zeros((B, table.n_det), dtype=np.int32)
    np.add.at(touch, (shot, table.u[e]), 1)
    vv = table.v[e]
    has_v = vv >= 0
    np.add.at(touch, (shot[has_v], vv[has_v]), 1)
    ok = touch[shot, table.u[e]] == 1
    ok &= np.where(has_v, touch[shot, np.maximum(vv, 0)] == 1, True)
    iso = np.zeros((B, E), dtype=bool)
    iso[shot[ok], e[ok]] = True
    return iso


def sample_with_faults(dem: stim.DetectorErrorModel, shots: int, seed: int):
    """(dets (B, n_det) bool, obs (B,) bool, errs (B, M) bool) from the DEM sampler."""
    dets, obs, errs = dem.compile_sampler(seed=seed).sample(shots, return_errors=True)
    return dets, obs[:, 0], errs
