"""Syndrome (fired detectors) -> batched k-nearest-neighbour graphs."""
from dataclasses import dataclass

import numpy as np
import stim
import torch

BOUNDARY_CLIP = 4.0


@dataclass
class GraphBatch:
    x: torch.Tensor           # (N, 8) node features
    edge_index: torch.Tensor  # (2, E) long, row 0 = source, row 1 = destination
    edge_attr: torch.Tensor   # (E, 4): dx, dy, dt, 1/|d|
    batch: torch.Tensor       # (N,) graph id of each node
    n_graphs: int
    nonempty: torch.Tensor    # (n_graphs,) bool
    det: torch.Tensor | None = None  # (N,) detector id of each node

    def to(self, device) -> "GraphBatch":
        return GraphBatch(self.x.to(device), self.edge_index.to(device), self.edge_attr.to(device),
                          self.batch.to(device), self.n_graphs, self.nonempty.to(device),
                          None if self.det is None else self.det.to(device))


def detector_table(circuit: stim.Circuit, d: int) -> np.ndarray:
    """(n_det, 4) array: x, y, t, is_x_type. X-type = ancilla that receives H gates."""
    coords = circuit.get_detector_coordinates()
    qcoords = {tuple(v): q for q, v in circuit.get_final_qubit_coordinates().items()}
    h_qubits = {t.value for op in circuit.flattened() if op.name == "H"
                for t in op.targets_copy()}
    rows = []
    for k in range(circuit.num_detectors):
        x, y, t = coords[k][:3]
        q = qcoords.get((x, y))
        rows.append((x, y, t, float(q in h_qubits)))
    return np.array(rows, dtype=np.float64)


def _node_features(table: np.ndarray, d: int) -> np.ndarray:
    """Per-detector features, all in raw lattice units (distance-agnostic)."""
    x, y, t, is_x = table.T
    lo, hi = 0.0, 2.0 * d          # ancilla coordinate range of the rotated code
    clip = lambda v: np.clip(v, 0, BOUNDARY_CLIP)
    return np.stack([
        1 - is_x, is_x,
        clip(x - lo), clip(hi - x), clip(y - lo), clip(hi - y),
        (t == t.min()).astype(float), (t == t.max()).astype(float),
    ], axis=1).astype(np.float32)


def build_graph_batch(dets: np.ndarray, table: np.ndarray, d: int, rounds: int,
                      k: int = 8) -> GraphBatch:
    feats_all = _node_features(table, d)
    B = dets.shape[0]
    shot, det = np.nonzero(dets)                     # sorted by shot
    counts = np.bincount(shot, minlength=B)
    nonempty = torch.from_numpy(counts > 0)
    x = torch.from_numpy(feats_all[det])
    batch = torch.from_numpy(shot.astype(np.int64))
    det_ids = torch.from_numpy(det.astype(np.int64))
    pos_flat = torch.from_numpy(table[det, :3].astype(np.float32))
    nmax = int(counts.max()) if B else 0
    if nmax < 2:
        empty = torch.zeros((2, 0), dtype=torch.long)
        return GraphBatch(x, empty, torch.zeros((0, 4)), batch, B, nonempty, det_ids)
    # Padded (B, nmax, 3) positions; position-in-graph of each node.
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    slot = torch.from_numpy(np.arange(len(det)) - starts[shot])
    pos = torch.zeros((B, nmax, 3))
    valid = torch.zeros((B, nmax), dtype=torch.bool)
    pos[batch, slot] = pos_flat
    valid[batch, slot] = True
    dist = torch.cdist(pos, pos)
    dist.masked_fill_(~valid[:, None, :], float("inf"))
    dist.diagonal(dim1=1, dim2=2).fill_(float("inf"))
    kk = min(k, nmax - 1)
    nd, nj = dist.topk(kk, dim=2, largest=False)     # (B, nmax, kk)
    ok = torch.isfinite(nd) & valid[:, :, None]
    b_idx, i_slot, _ = torch.nonzero(ok, as_tuple=True)
    j_slot = nj[ok]
    flat_index = torch.from_numpy(starts).long()
    dst = flat_index[b_idx] + i_slot
    src = flat_index[b_idx] + j_slot
    delta = pos_flat[src] - pos_flat[dst]
    inv = 1.0 / delta.norm(dim=1, keepdim=True)
    return GraphBatch(x, torch.stack([src, dst]), torch.cat([delta, inv], dim=1),
                      batch, B, nonempty, det_ids)
