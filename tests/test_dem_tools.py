import numpy as np
import pytest

from src.calibrate import matching_from_dem
from src.circuits import dem_of, noiseless_circuit
from src.dem_tools import parallel_edge_conflicts, robust_matching_from_dem
from src.evaluate import sample_shots
from src.noise import apply_noise, si1000_model
from src.scenarios import device_instance, load_scenarios


def _dems(d, p, scen):
    c = noiseless_circuit(d, d)
    t0, _ = device_instance(c, p, load_scenarios()[scen], 0)
    return c, dem_of(apply_noise(c, si1000_model(c, p))), apply_noise(c, t0)


def test_no_conflicts_without_crosstalk():
    for d in (3, 5):
        _, naive, _ = _dems(d, 0.003, "S0")
        assert parallel_edge_conflicts(naive) == {}


def test_crosstalk_d3_has_the_D1_boundary_conflict():
    _, _, true_c = _dems(3, 0.0025, "S4")
    conflicts = parallel_edge_conflicts(dem_of(true_c))
    assert (1, None) in conflicts
    assert conflicts[(1, None)] == {frozenset({0}), frozenset()}


def test_robust_matches_default_when_no_conflicts():
    _, naive, _ = _dems(3, 0.003, "S0")
    a = {(u, v): (frozenset(x["fault_ids"]), x["error_probability"]) for u, v, x in matching_from_dem(naive).edges()}
    b = {(u, v): (frozenset(x["fault_ids"]), x["error_probability"]) for u, v, x in robust_matching_from_dem(naive).edges()}
    assert a.keys() == b.keys()
    for k in a:
        assert a[k][0] == b[k][0] and a[k][1] == pytest.approx(b[k][1], rel=1e-6)


def test_robust_fixes_crosstalk_d3_oracle():
    c, naive, true_c = _dems(3, 0.0025, "S4")
    dets, obs = sample_shots(true_c, 200_000, seed=11)
    err = lambda m: float((m.decode_batch(dets)[:, 0] != obs).mean())
    default, robust = err(matching_from_dem(dem_of(true_c))), err(robust_matching_from_dem(dem_of(true_c)))
    assert robust < default - 0.005                      # clearly repaired
    assert robust <= err(matching_from_dem(naive)) * 1.03  # oracle no longer worse than naive
