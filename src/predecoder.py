"""Neural pre-decoder: scores matching-graph faults as isolated or not, so confident ones can be
removed from the syndrome before static matching decodes the rest."""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.dem_labels import EdgeTable
from src.gnn import _Block, _mlp
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


class PreDecoder(nn.Module):
    """Message-passing body as in GNNDecoder, with a symmetric pair head and a boundary head."""

    def __init__(self, node_dim: int = 8, edge_dim: int = 4, hidden: int = 64, layers: int = 4):
        super().__init__()
        self.node_enc = _mlp(node_dim, hidden, hidden)
        self.edge_enc = _mlp(edge_dim, hidden, hidden)
        self.blocks = nn.ModuleList(_Block(hidden) for _ in range(layers))
        self.pair_head = _mlp(2 * hidden + edge_dim, hidden, 1)
        self.bnd_head = _mlp(hidden, hidden, 1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, g: GraphBatch, c: Candidates) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.node_enc(g.x)
        e = self.edge_enc(g.edge_attr)
        for blk in self.blocks:
            h = blk(h, e, g.edge_index)
        hu, hv = h[c.pair_src], h[c.pair_dst]
        attr = g.edge_attr[c.pair_row]
        attr = torch.cat([attr[:, :3].abs(), attr[:, 3:]], dim=1)     # direction-free edge features
        pair_logits = self.pair_head(torch.cat([hu + hv, (hu - hv).abs(), attr], dim=1)).squeeze(1)
        bnd_logits = self.bnd_head(h[c.bnd_node]).squeeze(1)
        return pair_logits, bnd_logits


@dataclass
class RemovalResult:
    residual: np.ndarray        # (B, n_det) bool, syndrome after removals
    logical: np.ndarray         # (B,) bool, XOR of the removed faults' logical labels
    n_removed_dets: np.ndarray  # (B,) int
    n_fired: np.ndarray         # (B,) int, fired detectors before removal


def select_removals(shot: np.ndarray, det_a: np.ndarray, det_b: np.ndarray, prob: np.ndarray,
                    tau: float) -> np.ndarray:
    """Keep-mask over candidates: probability >= tau, resolved greedily per shot in descending
    probability so no detector is used twice. det_b == -1 marks a boundary candidate."""
    keep = np.zeros(len(shot), dtype=bool)
    idx = np.nonzero(prob >= tau)[0]
    if len(idx) == 0:
        return keep
    order = idx[np.lexsort((-prob[idx], shot[idx]))]
    used = set()
    for i in order.tolist():
        s = int(shot[i]) << 20
        ka = s + int(det_a[i])
        kb = s + int(det_b[i]) if det_b[i] >= 0 else -1
        if ka in used or (kb >= 0 and kb in used):
            continue
        used.add(ka)
        if kb >= 0:
            used.add(kb)
        keep[i] = True
    return keep


def predecode(model, table: EdgeTable, det_table: np.ndarray, dets: np.ndarray, d: int, tau: float,
              k: int = 8, batch: int = 4096, device: str = "cpu") -> RemovalResult:
    """Remove faults the model scores >= tau; return the residual syndrome and the logical contribution."""
    from src.graph import build_graph_batch
    B = dets.shape[0]
    residual = dets.copy()
    logical_count = np.zeros(B, dtype=np.int64)
    n_removed = np.zeros(B, dtype=np.int64)
    getattr(model, "eval", lambda: None)()
    with torch.no_grad():
        for s in range(0, B, batch):
            g = build_graph_batch(dets[s:s + batch], det_table, d=d, rounds=d, k=k).to(device)
            c = build_candidates(g, table)
            pair_logit, bnd_logit = model(g, c)
            node_shot, node_det = g.batch.cpu().numpy(), g.det.cpu().numpy()
            ps, pd_ = c.pair_src.cpu().numpy(), c.pair_dst.cpu().numpy()
            bn = c.bnd_node.cpu().numpy()
            shot = np.concatenate([node_shot[ps], node_shot[bn]])
            det_a = np.concatenate([node_det[ps], node_det[bn]])
            det_b = np.concatenate([node_det[pd_], -np.ones(len(bn), dtype=np.int64)])
            edge = np.concatenate([c.pair_edge.cpu().numpy(), c.bnd_edge.cpu().numpy()])
            prob = np.concatenate([torch.sigmoid(pair_logit).cpu().numpy(), torch.sigmoid(bnd_logit).cpu().numpy()])
            keep = select_removals(shot, det_a, det_b, prob, tau)
            rows, a, b, e = shot[keep] + s, det_a[keep], det_b[keep], edge[keep]
            residual[rows, a] = False
            has_b = b >= 0
            residual[rows[has_b], b[has_b]] = False
            np.add.at(logical_count, rows, table.label[e].astype(np.int64))
            np.add.at(n_removed, rows, 1 + has_b.astype(np.int64))
    return RemovalResult(residual, logical_count % 2 == 1, n_removed, dets.sum(axis=1))


def predecoded_errors(model, table: EdgeTable, det_table: np.ndarray, matching, dets: np.ndarray,
                      obs: np.ndarray, d: int, tau: float, return_result: bool = False, **kw):
    """Per-shot bool errors of predecode() followed by matching on the residual syndrome."""
    r = predecode(model, table, det_table, dets, d, tau, **kw)
    pred = matching.decode_batch(r.residual)[:, 0].astype(bool) ^ r.logical
    errors = pred != obs
    return (errors, r) if return_result else errors
