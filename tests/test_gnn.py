import numpy as np
import torch

from src.circuits import noiseless_circuit
from src.gnn import EMPTY_LOGIT, GNNDecoder
from src.graph import GraphBatch, build_graph_batch, detector_table


def _batch(d, counts=None, seed=0, p=0.08, B=12):
    t = detector_table(noiseless_circuit(d, d), d)
    rng = np.random.default_rng(seed)
    if counts is None:
        dets = rng.random((B, len(t))) < p
    else:
        dets = np.zeros((len(counts), len(t)), dtype=bool)
        for r, c in enumerate(counts):
            dets[r, rng.choice(len(t), size=c, replace=False)] = True
    return build_graph_batch(dets, t, d=d, rounds=d, k=8)


def test_output_shape():
    torch.manual_seed(0)
    assert GNNDecoder(hidden=16, layers=2)(_batch(3)).shape == (12,)


def test_empty_graphs_predict_no_flip():
    torch.manual_seed(0)
    g = _batch(3, counts=[0, 5, 0, 1, 2])
    out = GNNDecoder(hidden=16, layers=2)(g)
    assert torch.isfinite(out).all()
    assert (out[~g.nonempty] == EMPTY_LOGIT).all()
    assert (out[g.nonempty] != EMPTY_LOGIT).all()


def test_permutation_invariance():
    torch.manual_seed(0)
    model = GNNDecoder(hidden=16, layers=2).eval()
    g = _batch(5, seed=3)
    perm = torch.randperm(g.x.shape[0])
    inv = torch.empty_like(perm); inv[perm] = torch.arange(len(perm))
    g2 = GraphBatch(g.x[perm], inv[g.edge_index], g.edge_attr, g.batch[perm],
                    g.n_graphs, g.nonempty)
    assert torch.allclose(model(g), model(g2), atol=1e-5)


def test_runs_on_d3_and_d7():
    torch.manual_seed(0)
    model = GNNDecoder(hidden=16, layers=2)
    assert model(_batch(3)).shape == (12,)
    assert model(_batch(7, p=0.02)).shape == (12,)


def test_gradients_flow():
    torch.manual_seed(0)
    model = GNNDecoder(hidden=16, layers=2)
    out = model(_batch(5, seed=1))
    torch.nn.functional.binary_cross_entropy_with_logits(out, torch.ones_like(out)).backward()
    assert all(p.grad is not None for p in model.parameters())
    assert model.n_params() == sum(p.numel() for p in model.parameters())
