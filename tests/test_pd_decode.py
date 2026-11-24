import numpy as np
import torch

from src.circuits import dem_of, noiseless_circuit
from src.dem_labels import build_edge_table, fault_parity, isolated_faults, sample_with_faults
from src.dem_tools import robust_matching_from_dem
from src.evaluate import sample_shots
from src.graph import detector_table
from src.noise import apply_noise, si1000_model
from src.predecoder import PreDecoder, candidate_targets, predecode, predecoded_errors, select_removals


class _Stub:
    """A 'model' whose logits come from callables, so pipeline tests do not depend on training."""

    def __init__(self, pair_fn, bnd_fn):
        self.pair_fn, self.bnd_fn = pair_fn, bnd_fn

    def __call__(self, g, c):
        return self.pair_fn(g, c), self.bnd_fn(g, c)


def _const(pair, bnd):
    return _Stub(lambda g, c: torch.full((len(c.pair_src),), float(pair)),
                 lambda g, c: torch.full((len(c.bnd_node),), float(bnd)))


def _oracle(table, iso):
    def logits(g, c):
        py, by = candidate_targets(g, c, iso)
        return torch.where(py > 0.5, 20.0, -20.0), torch.where(by > 0.5, 20.0, -20.0)
    return _Stub(lambda g, c: logits(g, c)[0], lambda g, c: logits(g, c)[1])


def _setup(d, p=0.004):
    c = noiseless_circuit(d, d)
    noisy = apply_noise(c, si1000_model(c, p))
    dem = dem_of(noisy)
    return c, noisy, dem, build_edge_table(dem), detector_table(c, d), robust_matching_from_dem(dem)


def test_select_removals_resolves_conflicts():
    shot = np.array([0, 0, 0, 0, 1])
    a = np.array([5, 5, 1, 2, 5])
    b = np.array([-1, 6, 2, 3, -1])
    prob = np.array([0.95, 0.99, 0.9, 0.8, 0.97])
    assert select_removals(shot, a, b, prob, tau=0.9).tolist() == [False, True, True, False, True]
    # two pairs sharing detector 2: only the more probable survives
    keep = select_removals(np.zeros(2, int), np.array([1, 2]), np.array([2, 3]), np.array([0.6, 0.8]), tau=0.5)
    assert keep.tolist() == [False, True]
    assert select_removals(shot, a, b, prob, tau=1.01).sum() == 0


def test_tau_above_one_reproduces_matching():
    c, noisy, dem, table, dt, matching = _setup(3)
    dets, obs = sample_shots(noisy, 20_000, seed=11)
    torch.manual_seed(0)
    model = PreDecoder(hidden=8, layers=1)
    got = predecoded_errors(model, table, dt, matching, dets, obs, d=3, tau=1.01)
    assert np.array_equal(got, matching.decode_batch(dets)[:, 0].astype(bool) != obs)


def test_removed_detectors_flip_exactly_their_endpoints():
    c, noisy, dem, table, dt, matching = _setup(3)
    dets, obs, errs = sample_with_faults(dem, 3000, seed=5)
    iso = isolated_faults(table, fault_parity(table, errs))
    r = predecode(_oracle(table, iso), table, dt, dets, d=3, tau=0.5, batch=3000)
    assert not (r.residual & ~dets).any()                                    # only fired detectors removed
    assert np.array_equal((dets & ~r.residual).sum(axis=1), r.n_removed_dets)  # each detector removed once
    assert np.array_equal(r.n_fired, dets.sum(axis=1))
    assert r.n_removed_dets.sum() > 0


def test_oracle_removal_is_not_worse_than_matching():
    c, noisy, dem, table, dt, matching = _setup(3)
    dets, obs, errs = sample_with_faults(dem, 50_000, seed=6)
    iso = isolated_faults(table, fault_parity(table, errs))
    pd = predecoded_errors(_oracle(table, iso), table, dt, matching, dets, obs, d=3, tau=0.5, batch=50_000)
    base = matching.decode_batch(dets)[:, 0].astype(bool) != obs
    assert pd.sum() <= 1.05 * base.sum() + 20, (pd.sum(), base.sum())


def test_all_removed_gives_empty_residual():
    c, noisy, dem, table, dt, matching = _setup(3)
    a, b = next((a, b) for a in range(table.n_det) for b in range(a + 1, table.n_det) if table.edge_id[a, b] >= 0)
    dets = np.zeros((1, len(dt)), dtype=bool)
    dets[0, [a, b]] = True
    obs = np.array([bool(table.label[table.edge_id[a, b]])])
    r = predecode(_const(20, -20), table, dt, dets, d=3, tau=0.5)
    assert not r.residual.any() and r.n_removed_dets[0] == 2
    err = predecoded_errors(_const(20, -20), table, dt, matching, dets, obs, d=3, tau=0.5)
    assert not err.any()


def test_empty_and_single_defect_batches():
    c, noisy, dem, table, dt, matching = _setup(3)
    dets = np.zeros((3, len(dt)), dtype=bool)
    dets[1, 5] = True
    r = predecode(_const(20, 20), table, dt, dets, d=3, tau=0.5)
    assert r.n_removed_dets[0] == 0 and r.n_removed_dets[2] == 0 and not r.logical[0] and not r.logical[2]
    assert r.n_removed_dets[1] == 1 and not r.residual.any()          # lone defect leaves via its boundary edge
    none = predecode(_const(20, 20), table, dt, dets, d=3, tau=1.01)
    assert none.n_removed_dets.sum() == 0 and np.array_equal(none.residual, dets)


def test_score_once_apply_many_taus_matches_predecode():
    from src.predecoder import apply_tau, score
    c, noisy, dem, table, dt, matching = _setup(3)
    dets, obs, errs = sample_with_faults(dem, 4000, seed=8)
    torch.manual_seed(1)
    model = PreDecoder(hidden=8, layers=1)
    sc = score(model, table, dt, dets, d=3, batch=1500)
    for tau in (0.0, 0.4, 0.5, 0.9, 1.01):
        a = apply_tau(sc, table, dets, tau)
        b = predecode(model, table, dt, dets, d=3, tau=tau, batch=1500)
        assert np.array_equal(a.residual, b.residual) and np.array_equal(a.logical, b.logical)
        assert np.array_equal(a.n_removed_dets, b.n_removed_dets)


def test_removed_faults_are_recorded():
    c, noisy, dem, table, dt, matching = _setup(3)
    dets, obs, errs = sample_with_faults(dem, 3000, seed=5)
    iso = isolated_faults(table, fault_parity(table, errs))
    r = predecode(_oracle(table, iso), table, dt, dets, d=3, tau=0.5, batch=3000)
    assert len(r.removed_shot) == len(r.removed_edge) > 0
    ends = 1 + (table.v[r.removed_edge] >= 0)
    assert int(ends.sum()) == int(r.n_removed_dets.sum())
    assert iso[r.removed_shot, r.removed_edge].all()                    # the oracle only removes true isolated faults
