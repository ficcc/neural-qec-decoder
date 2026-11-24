"""DEM -> matching-graph helpers that keep logical-flip labels correct.

A matching graph holds one logical-flip label per edge. When a detector error model has several
mechanisms on the same detector pair with *different* logical effects (e.g. crosstalk adds
`error(p) D1` next to the existing `error(q) D1 L0`), PyMatching's default merge can keep the
wrong label. `robust_matching_from_dem` merges such mechanisms itself: probabilities combine as
independent flips, the label is that of the most probable mechanism.
"""
from collections import defaultdict

import numpy as np
import pymatching
import stim

P_MIN, P_MAX = 1e-12, 0.5 - 1e-12


def _components(dem: stim.DetectorErrorModel):
    """Yield (p, node_u, node_v_or_None, frozenset(observables)) for every graph-like component."""
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        comp_dets, comp_obs = [], set()
        for t in list(inst.targets_copy()) + [stim.target_separator()]:
            if t.is_separator():
                if comp_dets:
                    if len(comp_dets) > 2:
                        raise ValueError(f"non-graphlike component with detectors {comp_dets}")
                    u = comp_dets[0]; v = comp_dets[1] if len(comp_dets) == 2 else None
                    yield p, min(u, v) if v is not None else u, (max(u, v) if v is not None else None), frozenset(comp_obs)
                comp_dets, comp_obs = [], set()
            elif t.is_relative_detector_id():
                comp_dets.append(t.val)
            elif t.is_logical_observable_id():
                comp_obs.add(t.val)


def _group(dem):
    groups = defaultdict(list)
    for p, u, v, obs in _components(dem):
        groups[(u, v)].append((p, obs))
    return groups


def parallel_edge_conflicts(dem: stim.DetectorErrorModel) -> dict:
    """{(u, v): {labels}} for matching edges whose mechanisms disagree on the logical-flip label."""
    return {k: {obs for _, obs in v} for k, v in _group(dem).items() if len({obs for _, obs in v}) > 1}


def robust_matching_from_dem(dem: stim.DetectorErrorModel) -> pymatching.Matching:
    m = pymatching.Matching()
    for (u, v), mechs in sorted(_group(dem).items(), key=lambda kv: (kv[0][0], -1 if kv[0][1] is None else kv[0][1])):
        p_tot = 0.0
        for p, _ in mechs:
            p_tot = p_tot * (1 - p) + p * (1 - p_tot)
        p_tot = float(np.clip(p_tot, P_MIN, P_MAX))
        label = max(mechs, key=lambda x: x[0])[1]
        w = float(np.log((1 - p_tot) / p_tot))
        if v is None:
            m.add_boundary_edge(u, fault_ids=set(label), weight=w, error_probability=p_tot)
        else:
            m.add_edge(u, v, fault_ids=set(label), weight=w, error_probability=p_tot)
    return m
