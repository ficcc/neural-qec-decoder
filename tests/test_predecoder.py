from dataclasses import replace

import numpy as np
import pytest
import torch

from src.circuits import dem_of, noiseless_circuit
from src.dem_labels import build_edge_table, fault_parity, isolated_faults, sample_with_faults
from src.graph import build_graph_batch, detector_table
from src.noise import apply_noise, si1000_model
from src.predecoder import build_candidates, candidate_targets


def _setup(d, p=0.004):
    c = noiseless_circuit(d, d)
    dem = dem_of(apply_noise(c, si1000_model(c, p)))
    return build_edge_table(dem), detector_table(c, d), dem


def _graph(dets, det_table, d, k=8):
    return build_graph_batch(dets, det_table, d=d, rounds=d, k=k)


def test_pairs_are_knn_edges_and_matching_edges():
    table, dt, _ = _setup(3)
    dets = np.random.default_rng(0).random((16, len(dt))) < 0.15
    g = _graph(dets, dt, 3)
    c = build_candidates(g, table)
    assert len(c.pair_src) > 0
    s, t = c.pair_src.numpy(), c.pair_dst.numpy()
    assert (s < t).all()
    assert len(set(zip(s.tolist(), t.tolist()))) == len(s)
    assert torch.equal(g.batch[c.pair_src], g.batch[c.pair_dst])
    ea, eb = g.det[c.pair_src].numpy(), g.det[c.pair_dst].numpy()
    assert (table.edge_id[ea, eb] == c.pair_edge.numpy()).all() and (c.pair_edge >= 0).all()
    rows = g.edge_index[:, c.pair_row]
    assert (((rows[0] == c.pair_src) & (rows[1] == c.pair_dst)) | ((rows[0] == c.pair_dst) & (rows[1] == c.pair_src))).all()


def test_non_matching_knn_edge_is_not_a_candidate():
    table, dt, _ = _setup(3)
    a, b = next((a, b) for a in range(table.n_det) for b in range(a + 1, table.n_det) if table.edge_id[a, b] < 0)
    dets = np.zeros((1, len(dt)), dtype=bool)
    dets[0, [a, b]] = True
    g = _graph(dets, dt, 3)
    assert g.edge_index.shape[1] == 2                      # the kNN graph does connect them...
    assert len(build_candidates(g, table).pair_src) == 0   # ...but they are not a matching edge


def test_boundary_candidates_only_for_boundary_detectors():
    # At d=3 every detector has a boundary edge (decomposition creates single-detector
    # pieces), so the no-boundary-edge branch is exercised on a table with one edge removed.
    table, dt, _ = _setup(3)
    det = 5
    assert table.bnd_id[det] >= 0
    dets = np.zeros((1, len(dt)), dtype=bool)
    dets[0, det] = True
    c = build_candidates(_graph(dets, dt, 3), table)
    assert len(c.bnd_node) == 1 and len(c.pair_src) == 0 and int(c.bnd_edge[0]) == int(table.bnd_id[det])
    no_b = table.bnd_id.copy()
    no_b[det] = -1
    c = build_candidates(_graph(dets, dt, 3), replace(table, bnd_id=no_b))
    assert len(c.bnd_node) == 0


def test_empty_and_single_defect_syndromes():
    table, dt, _ = _setup(3)
    c = build_candidates(_graph(np.zeros((4, len(dt)), dtype=bool), dt, 3), table)
    assert len(c.pair_src) == len(c.pair_dst) == len(c.pair_edge) == len(c.pair_row) == len(c.bnd_node) == 0
    mixed = np.zeros((3, len(dt)), dtype=bool)
    mixed[1, int(np.nonzero(table.bnd_id >= 0)[0][0])] = True
    c = build_candidates(_graph(mixed, dt, 3), table)
    assert len(c.pair_src) == 0 and len(c.bnd_node) == 1


def test_targets_align_with_isolated_labels_and_recall():
    table, dt, dem = _setup(5)
    dets, obs, errs = sample_with_faults(dem, 4000, seed=3)
    iso = isolated_faults(table, fault_parity(table, errs))
    g = _graph(dets, dt, 5)
    c = build_candidates(g, table)
    pair_y, bnd_y = candidate_targets(g, c, iso)
    assert pair_y.shape == c.pair_src.shape and bnd_y.shape == c.bnd_node.shape
    expected = iso[g.batch[c.pair_src].numpy(), c.pair_edge.numpy()]
    assert np.array_equal(pair_y.numpy().astype(bool), expected)
    recall = (pair_y.sum() + bnd_y.sum()).item() / iso.sum()
    assert recall >= 0.98, recall
