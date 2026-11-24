import pytest

from experiments.e8_warmstart import steps_to_reach


def _h(*pairs):
    return [{"step": s, "val_ler": v} for s, v in pairs]


def test_steps_to_reach_first_crossing():
    assert steps_to_reach(_h((0, 0.3), (500, 0.2), (1000, 0.14), (1500, 0.12)), 0.14) == 1000


def test_steps_to_reach_none_if_never():
    assert steps_to_reach(_h((0, 0.3), (500, 0.2)), 0.1) is None
