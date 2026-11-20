import numpy as np
import torch

from src.circuits import noiseless_circuit
from src.graph import build_graph_batch, detector_table


def _table(d):
    return detector_table(noiseless_circuit(d, d), d)


def _dets_with_counts(n_det, counts, seed=0):
    rng = np.random.default_rng(seed)
    dets = np.zeros((len(counts), n_det), dtype=bool)
    for r, c in enumerate(counts):
        dets[r, rng.choice(n_det, size=c, replace=False)] = True
    return dets


def test_shapes():
    t = _table(3)
    dets = np.random.default_rng(0).random((16, len(t))) < 0.1
    g = build_graph_batch(dets, t, d=3, rounds=3, k=8)
    assert g.x.shape == (dets.sum(), 8)
    assert g.edge_attr.shape[1] == 4 and g.edge_index.shape[0] == 2
    assert g.batch.min() >= 0 and g.batch.max() < 16 and g.n_graphs == 16


def test_empty_and_nonempty_mixed():
    t = _table(3)
    dets = _dets_with_counts(len(t), [0, 5, 0, 3])
    g = build_graph_batch(dets, t, d=3, rounds=3, k=8)
    assert g.nonempty.tolist() == [False, True, False, True]
    assert g.x.shape[0] == 8


def test_tiny_graphs():
    t = _table(3)
    g = build_graph_batch(_dets_with_counts(len(t), [1]), t, d=3, rounds=3, k=8)
    assert g.edge_index.shape[1] == 0
    g = build_graph_batch(_dets_with_counts(len(t), [2]), t, d=3, rounds=3, k=8)
    assert g.edge_index.shape[1] == 2


def test_knn_degree():
    t = _table(5)
    g = build_graph_batch(_dets_with_counts(len(t), [20]), t, d=5, rounds=5, k=8)
    assert torch.bincount(g.edge_index[1]).tolist() == [8] * 20


def test_edges_stay_in_graph():
    t = _table(5)
    dets = np.random.default_rng(1).random((32, len(t))) < 0.05
    g = build_graph_batch(dets, t, d=5, rounds=5, k=8)
    assert torch.equal(g.batch[g.edge_index[0]], g.batch[g.edge_index[1]])
    assert (g.edge_attr[:, 3] > 0).all()


def test_distance_agnostic_features():
    # Detector one lattice unit right of the left ancilla edge, mid-round, same type.
    feats = {}
    for d in (3, 7):
        t = _table(d)
        idx = np.where((t[:, 0] == 2) & (t[:, 2] == 1) & (t[:, 1] == 2))[0][0]
        dets = np.zeros((1, len(t)), dtype=bool); dets[0, idx] = True
        feats[d] = build_graph_batch(dets, t, d=d, rounds=d, k=8).x[0]
    assert feats[3][2] == feats[7][2]   # left-boundary distance
    assert feats[3][4] == feats[7][4]   # bottom-boundary distance
    assert feats[3][0:2].tolist() == feats[7][0:2].tolist()


def test_type_split():
    t = _table(3)
    n_x, n_z = int(t[:, 3].sum()), int((1 - t[:, 3]).sum())
    # d=3, 3 rounds: 4 X-ancillas x 2 rounds = 8 X dets; 4 Z x 3 rounds + 4 final = 16 Z dets.
    assert (n_x, n_z) == (8, 16)
