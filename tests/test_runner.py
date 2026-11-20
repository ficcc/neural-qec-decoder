import numpy as np

from src.circuits import noiseless_circuit
from src.evaluate import sample_shots
from src.runner import gnn_decoder, matching_decoders
from src.scenarios import device_instance, load_scenarios


def test_s0_naive_equals_oracle_and_all_present():
    c = noiseless_circuit(3, 3)
    t0, t1 = device_instance(c, 0.006, load_scenarios()["S0"], 0)
    decs, test_c = matching_decoders(c, 0.006, t0, t1, cal_shots=20_000, cal_seed=5)
    assert set(decs) == {"naive", "oracle", "calibrated", "correlated"}
    dets, obs = sample_shots(test_c, 20_000, seed=9)
    assert decs["naive"](dets, obs) == decs["oracle"](dets, obs)
    assert decs["correlated"](dets, obs) <= decs["naive"](dets, obs) * 1.2 + 5


def test_s3_test_circuit_uses_t1():
    c = noiseless_circuit(3, 3)
    t0, t1 = device_instance(c, 0.004, load_scenarios()["S3"], 0)
    _, test_c = matching_decoders(c, 0.004, t0, t1, cal_shots=1000, cal_seed=5)
    from src.noise import apply_noise
    assert test_c == apply_noise(c, t1) and test_c != apply_noise(c, t0)


def test_gnn_decoder_counts_errors():
    import torch
    from src.gnn import GNNDecoder
    torch.manual_seed(0)
    f = gnn_decoder(GNNDecoder(hidden=8, layers=1), d=3, k=8)
    dets = np.zeros((10, 24), dtype=bool); obs = np.ones(10, dtype=bool)
    assert f(dets, obs) == 10   # empty syndromes predict "no flip"
