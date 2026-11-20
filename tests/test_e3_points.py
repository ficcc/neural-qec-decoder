from experiments.e3_generalise import points
from experiments.jobs import E3_TEST_POOL, E3_TRAIN_POOL


def test_e3_test_devices_unseen_in_training():
    s1 = {seed for scen, seed, _ in points() if scen == "S1"}
    assert s1 == set(E3_TEST_POOL) and not s1 & set(E3_TRAIN_POOL)
