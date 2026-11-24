"""Single source of truth for every GNN training job in E1-E4 (name -> TrainConfig).

Experiment scripts load checkpoints from ckpt_dir(name); train_jobs.py trains them,
so training can run on a GPU pod while evaluation runs locally.
"""
from dataclasses import replace
from pathlib import Path

from src.train import TrainConfig

E1_GRID = (0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.010)  # SI1000 p_th ~0.008
E2_TRAIN_GRID = (0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007)       # up to ~0.9 p_th
E2_TEST_P = (0.002, 0.004)                                               # ~0.25, 0.5 p_th
E3_TRAIN_POOL = tuple(5000 + i for i in range(20))
E3_TEST_POOL = tuple(9000 + i for i in range(3))
E4_P = 0.004
E4_BUDGETS = (1_000, 10_000, 100_000, 1_000_000)
E6_TRAIN_GRID = (0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035)          # S4 p_th ~0.0035-0.004
E6_TEST_P = (0.0015, 0.0025)
DISTANCES = (3, 5, 7)
SEEDS3 = (0, 1, 2)


def ckpt_dir(name: str) -> Path:
    return Path("checkpoints") / name


def all_jobs() -> list[tuple[str, TrainConfig]]:
    jobs = []
    for d in DISTANCES:                                                     # E1: S0 control
        for s in SEEDS3:
            jobs.append((f"e1/d{d}_s{s}", TrainConfig("S0", 0, (d,), E1_GRID, seed=s)))
    for scen in ("S1", "S2", "S3"):                                         # E2: mismatch
        for inst in (0, 1, 2):
            for d in DISTANCES:
                jobs.append((f"e2/{scen}_i{inst}_d{d}_s0", TrainConfig(scen, inst, (d,), E2_TRAIN_GRID, seed=0)))
    for d in DISTANCES:                                                     # E2: seed spread
        for s in (1, 2):
            jobs.append((f"e2/S2_i0_d{d}_s{s}", TrainConfig("S2", 0, (d,), E2_TRAIN_GRID, seed=s)))
    for s in SEEDS3:                                                        # E3: generalisation
        jobs.append((f"e3/S0_d35_s{s}", TrainConfig("S0", 0, (3, 5), E1_GRID, seed=s)))
    jobs.append(("e3/S1_d35_pool", TrainConfig("S1", 0, (3, 5), E2_TRAIN_GRID, instance_pool=E3_TRAIN_POOL)))
    jobs.append(("e3/S1_d7_pool", TrainConfig("S1", 0, (7,), E2_TRAIN_GRID, instance_pool=E3_TRAIN_POOL)))
    for b in E4_BUDGETS:                                                    # E4: data budget
        for s in SEEDS3:
            jobs.append((f"e4/B{b}_s{s}", TrainConfig("S2", 0, (5,), (E4_P,), seed=s, fixed_shots=b)))
    for d in DISTANCES:                                                     # E6: crosstalk
        for s in ((0, 1, 2) if d == 5 else (0,)):
            jobs.append((f"e6/S4_d{d}_s{s}", TrainConfig("S4", 0, (d,), E6_TRAIN_GRID, seed=s)))
    return jobs


def ablation_jobs() -> list[tuple[str, TrainConfig]]:
    """Receptive-field / capacity ablation at d=7 (blueprint 'Test A' mechanism check)."""
    return [("abl/d7_L8_H96_s0", TrainConfig("S0", 0, (7,), E1_GRID, seed=0, layers=8, hidden=96))]


def warmstart_jobs() -> list[tuple[str, TrainConfig]]:
    """d=7 GNNs initialised from smaller-distance models; every other setting (grid, steps, lr, seed,
    architecture) is identical to the from-scratch E1 d=7 job of the same seed, so only init differs."""
    scratch = dict(all_jobs())
    jobs = []
    for s in SEEDS3:
        src = f"e3/S0_d35_s{s}"      # trained jointly on d=3 and d=5
        jobs.append((f"ws/d7_from_d35_s{s}", replace(scratch[f"e1/d7_s{s}"], init_from=str(ckpt_dir(src) / "best.pt"))))
    jobs.append(("ws/d7_from_d5_s0", replace(scratch["e1/d7_s0"], init_from=str(ckpt_dir("e1/d5_s0") / "best.pt"))))
    return jobs
