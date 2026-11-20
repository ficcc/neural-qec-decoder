"""Glue shared by experiment scripts: build decoders as (dets, obs) -> error-count callables."""
from typing import Callable

import numpy as np
import pymatching
import stim

from src.calibrate import calibrated_matching, matching_from_dem
from src.circuits import dem_of, noiseless_circuit
from src.evaluate import count_errors_gnn, count_errors_matching
from src.graph import detector_table
from src.noise import NoiseModel, apply_noise, uniform_model

Decoder = Callable[[np.ndarray, np.ndarray], int]


def _m(matching: pymatching.Matching, correlations: bool = False) -> Decoder:
    if correlations:
        return lambda dets, obs: int((matching.decode_batch(dets, enable_correlations=True)[:, 0] != obs).sum())
    return lambda dets, obs: count_errors_matching(matching, dets, obs)


def matching_decoders(c: stim.Circuit, p: float, t0: NoiseModel, t1: NoiseModel, *,
                      cal_shots: int, cal_seed: int) -> tuple[dict[str, Decoder], stim.Circuit]:
    """naive (uniform p), oracle (true t1 DEM), calibrated (p_ij on t0 data),
    correlated (correlated matching on the uniform DEM). Returns decoders and the t1 test circuit."""
    naive_dem = dem_of(apply_noise(c, uniform_model(c, p)))
    t0_c, t1_c = apply_noise(c, t0), apply_noise(c, t1)
    cal_dets = t0_c.compile_detector_sampler(seed=cal_seed).sample(cal_shots)
    decs = {
        "naive": _m(matching_from_dem(naive_dem)),
        "oracle": _m(matching_from_dem(dem_of(t1_c))),
        "calibrated": _m(calibrated_matching(naive_dem, cal_dets)),
        "correlated": _m(pymatching.Matching.from_detector_error_model(naive_dem, enable_correlations=True), True),
    }
    return decs, t1_c


def gnn_decoder(model, d: int, k: int) -> Decoder:
    table = detector_table(noiseless_circuit(d, d), d)
    return lambda dets, obs: count_errors_gnn(model, dets, obs, table, d=d, rounds=d, k=k)
