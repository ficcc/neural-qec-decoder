"""Neural pre-decoder: scores matching-graph faults as isolated or not, so confident ones can be
removed from the syndrome before static matching decodes the rest."""
from dataclasses import dataclass

import numpy as np
import torch

from src.dem_labels import EdgeTable
from src.graph import GraphBatch


@dataclass
class Candidates:
    pair_src: torch.Tensor   # (P,) global node index, pair_src < pair_dst, each unordered pair once
    pair_dst: torch.Tensor   # (P,)
    pair_edge: torch.Tensor  # (P,) edge id in the EdgeTable
    pair_row: torch.Tensor   # (P,) a row of g.edge_index / g.edge_attr carrying this pair
    bnd_node: torch.Tensor   # (Q,) global node index of a fired detector with a boundary edge
    bnd_edge: torch.Tensor   # (Q,) edge id of that boundary edge


def build_candidates(g: GraphBatch, table: EdgeTable) -> Candidates:
    """Pair candidates: kNN edges whose detector pair is a matching edge. Boundary candidates:
    fired detectors that have a boundary edge."""
    dev = g.x.device
    n_nodes = max(int(g.x.shape[0]), 1)
    src, dst = g.edge_index
    edge_id = torch.from_numpy(table.edge_id).to(dev).long()
    eid = edge_id[g.det[src], g.det[dst]]
    rows = torch.nonzero(eid >= 0).squeeze(1)
    s, t = src[rows], dst[rows]
    key = torch.minimum(s, t) * n_nodes + torch.maximum(s, t)
    uniq, inverse = torch.unique(key, return_inverse=True)
    first = torch.full((len(uniq),), len(rows), dtype=torch.long, device=dev)
    first.scatter_reduce_(0, inverse, torch.arange(len(rows), device=dev), reduce="amin", include_self=True)
    rep = rows[first] if len(uniq) else rows[:0]
    bnd_id = torch.from_numpy(table.bnd_id).to(dev).long()[g.det]
    nodes = torch.nonzero(bnd_id >= 0).squeeze(1)
    return Candidates(pair_src=uniq // n_nodes, pair_dst=uniq % n_nodes, pair_edge=eid[rep], pair_row=rep,
                      bnd_node=nodes, bnd_edge=bnd_id[nodes])


def candidate_targets(g: GraphBatch, c: Candidates, isolated: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Float 0/1 targets: is the candidate's fault isolated? `isolated` is the (B, E) bool array."""
    iso = torch.from_numpy(isolated).to(g.x.device)
    pair_y = iso[g.batch[c.pair_src], c.pair_edge].float()
    bnd_y = iso[g.batch[c.bnd_node], c.bnd_edge].float()
    return pair_y, bnd_y
