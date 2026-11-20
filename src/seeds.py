"""Disjoint RNG seed ranges for training, validation and test samplers."""
SEEDS = {"train_base": 0, "val": 1_000_003, "test": 2_000_003}


def test_seed(chunk: int) -> int:
    """Seed for the i-th test chunk; never overlaps training or validation."""
    return SEEDS["test"] + chunk


test_seed.__test__ = False  # keep pytest from collecting it
