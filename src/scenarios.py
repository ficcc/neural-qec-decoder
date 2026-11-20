"""Noise scenarios S0-S3 and seeded "device instances" (per-qubit/gate rate draws)."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stim

from src.noise import NoiseModel, uniform_model

P_CAP = 0.25
FIELDS = ("p_1q", "p_2q", "p_reset", "p_meas", "p_data")


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    sigma: float      # log-normal spread of per-qubit/gate rates
    n_bad: int        # number of bad CX couplers
    bad_mult: float   # rate multiplier on bad couplers
    delta: float      # log-normal drift between calibration (t0) and test (t1)


def load_scenarios(path: str | Path = "configs/scenarios.json") -> dict[str, ScenarioConfig]:
    raw = json.loads(Path(path).read_text())
    return {k: ScenarioConfig(name=k, **v) for k, v in raw.items()}


def _finish(rates: dict[str, np.ndarray], p: float) -> dict[str, np.ndarray]:
    """Renormalise each channel to mean p, then clip to P_CAP."""
    out = {}
    for f, v in rates.items():
        v = v * (p / v.mean()) if v.mean() > 0 else v
        out[f] = np.minimum(v, P_CAP)
    return out


def _to_model(template: NoiseModel, rates: dict[str, np.ndarray]) -> NoiseModel:
    return NoiseModel(**{f: dict(zip(getattr(template, f), map(float, rates[f])))
                         for f in FIELDS})


def device_instance(circuit: stim.Circuit, p: float, cfg: ScenarioConfig,
                    seed: int) -> tuple[NoiseModel, NoiseModel]:
    """(t0 model for calibration/training, t1 model for testing).

    The random pattern depends only on (circuit, cfg, seed), not on p.
    """
    template = uniform_model(circuit, 1.0)
    rng = np.random.default_rng(seed)
    z = {f: rng.standard_normal(len(getattr(template, f))) for f in FIELDS}
    rates = {f: p * np.exp(cfg.sigma * z[f]) for f in FIELDS}
    if cfg.n_bad:
        bad = rng.choice(len(template.p_2q), size=cfg.n_bad, replace=False)
        rates["p_2q"][bad] *= cfg.bad_mult
    t0 = _finish(rates, p)
    if cfg.delta > 0:
        z1 = {f: rng.standard_normal(len(getattr(template, f))) for f in FIELDS}
        t1 = _finish({f: t0[f] * np.exp(cfg.delta * z1[f]) for f in FIELDS}, p)
    else:
        t1 = t0
    return _to_model(template, t0), _to_model(template, t1)
