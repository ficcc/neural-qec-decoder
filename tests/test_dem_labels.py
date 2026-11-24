import numpy as np

from src.circuits import dem_of, noiseless_circuit
from src.dem_labels import build_edge_table, fault_parity, isolated_faults, sample_with_faults
from src.dem_tools import robust_matching_from_dem
from src.noise import apply_noise, si1000_model
from src.scenarios import device_instance, load_scenarios


def _dem(d, p):
    c = noiseless_circuit(d, d)
    return dem_of(apply_noise(c, si1000_model(c, p)))


def test_edges_and_labels_match_robust_matching():
    for d in (3, 5):
        dem = _dem(d, 0.004)
        t = build_edge_table(dem)
        m = {(u, v): int(0 in a["fault_ids"]) for u, v, a in robust_matching_from_dem(dem).edges()}
        mine = {(int(a), None if b < 0 else int(b)): int(l) for a, b, l in zip(t.u, t.v, t.label)}
        assert mine == m
        assert t.edge_id.shape == (t.n_det, t.n_det)


def test_parity_reproduces_syndrome_and_logical():
    for d in (3, 5):
        dem = _dem(d, 0.004)
        t = build_edge_table(dem)
        dets, obs, errs = sample_with_faults(dem, 2000, seed=1)
        par = fault_parity(t, errs)
        assert par.shape == (2000, len(t.u))
        rebuilt = np.zeros(dets.shape, dtype=np.uint8)
        for e in range(len(t.u)):
            rows = par[:, e].astype(bool)
            rebuilt[rows, t.u[e]] ^= 1
            if t.v[e] >= 0:
                rebuilt[rows, t.v[e]] ^= 1
        assert np.array_equal(rebuilt.astype(bool), dets)
        assert np.array_equal((par.astype(np.int64) @ t.label.astype(np.int64)) % 2 == 1, obs)


def test_isolated_properties():
    dem = _dem(3, 0.0005)
    t = build_edge_table(dem)
    dets, obs, errs = sample_with_faults(dem, 20000, seed=2)
    par = fault_parity(t, errs)
    iso = isolated_faults(t, par)
    assert not (iso & (par == 0)).any()
    touch = np.zeros(dets.shape, dtype=np.int32)
    for e in range(len(t.u)):
        rows = par[:, e].astype(bool)
        touch[rows, t.u[e]] += 1
        if t.v[e] >= 0:
            touch[rows, t.v[e]] += 1
    for e in np.nonzero(iso.any(axis=0))[0]:
        rows = iso[:, e]
        assert (touch[rows, t.u[e]] == 1).all()
        if t.v[e] >= 0:
            assert (touch[rows, t.v[e]] == 1).all()
    assert iso.sum() > 0.95 * (par == 1).sum()


def test_parallel_edge_label_uses_dominant_mechanism():
    c = noiseless_circuit(3, 3)
    t0, _ = device_instance(c, 0.0025, load_scenarios()["S4"], 0)
    t = build_edge_table(dem_of(apply_noise(c, t0)))
    e = t.bnd_id[1]
    assert e >= 0 and t.label[e] == 1


def test_sampler_shapes_and_seed():
    dem = _dem(3, 0.004)
    a = sample_with_faults(dem, 500, seed=7)
    b = sample_with_faults(dem, 500, seed=7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert a[2].shape == (500, dem.num_errors) and a[0].shape == (500, dem.num_detectors) and a[1].shape == (500,)
