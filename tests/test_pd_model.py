from dataclasses import replace

import numpy as np
import torch

from src.circuits import dem_of, noiseless_circuit
from src.dem_labels import build_edge_table
from src.graph import GraphBatch, build_graph_batch, detector_table
from src.noise import apply_noise, si1000_model
from src.predecoder import PreDecoder, build_candidates


def _batch(d, density=0.12, shots=10, seed=0):
    c = noiseless_circuit(d, d)
    table = build_edge_table(dem_of(apply_noise(c, si1000_model(c, 0.004))))
    dt = detector_table(c, d)
    dets = np.random.default_rng(seed).random((shots, len(dt))) < density
    g = build_graph_batch(dets, dt, d=d, rounds=d, k=8)
    return g, build_candidates(g, table)


def _model():
    torch.manual_seed(0)
    return PreDecoder(hidden=16, layers=2).eval()


def test_output_shapes_and_finite():
    g, c = _batch(3)
    p, b = _model()(g, c)
    assert p.shape == c.pair_src.shape and b.shape == c.bnd_node.shape
    assert torch.isfinite(p).all() and torch.isfinite(b).all() and len(p) > 0


def test_pair_logit_is_symmetric_in_endpoints():
    g, c = _batch(3)
    m = _model()
    p1, _ = m(g, c)
    p2, _ = m(g, replace(c, pair_src=c.pair_dst, pair_dst=c.pair_src))
    assert torch.allclose(p1, p2, atol=1e-6)


def test_runs_with_no_candidates():
    g, c = _batch(3, density=0.0)
    p, b = _model()(g, c)
    assert p.numel() == 0 and b.numel() == 0


def test_runs_at_d3_and_d7():
    m = _model()
    for d, dens in ((3, 0.12), (7, 0.03)):
        g, c = _batch(d, dens)
        p, b = m(g, c)
        assert p.shape == c.pair_src.shape and torch.isfinite(p).all()


def test_permutation_invariance():
    g, c = _batch(5, density=0.06, seed=3)
    m = _model()
    p1, b1 = m(g, c)
    perm = torch.randperm(g.x.shape[0])
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(len(perm))
    g2 = GraphBatch(g.x[perm], inv[g.edge_index], g.edge_attr, g.batch[perm], g.n_graphs, g.nonempty, g.det[perm])
    c2 = replace(c, pair_src=inv[c.pair_src], pair_dst=inv[c.pair_dst], bnd_node=inv[c.bnd_node])
    p2, b2 = m(g2, c2)
    assert torch.allclose(p1, p2, atol=1e-5) and torch.allclose(b1, b2, atol=1e-5)


def test_gradients_flow():
    g, c = _batch(5, density=0.06)
    m = PreDecoder(hidden=16, layers=2)
    p, b = m(g, c)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(torch.cat([p, b]), torch.ones(len(p) + len(b)))
    loss.backward()
    assert all(q.grad is not None for q in m.parameters())
    assert m.n_params() == sum(q.numel() for q in m.parameters())
