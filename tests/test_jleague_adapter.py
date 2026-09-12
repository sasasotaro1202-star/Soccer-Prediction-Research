import pandas as pd

from src.data.jleague_adapter import _norm_comp, TARGETS


def test_jleague_competition_normalization_is_conservative():
    assert _norm_comp("J1") == "J1"
    assert _norm_comp("J1 League") == "J1"
    assert _norm_comp("J2") == "J2"
    assert _norm_comp("J3 League") == "J3"
    assert _norm_comp("Emperor Cup") is None


def test_jleague_targets_are_exactly_three():
    assert TARGETS == {"J1": "J1", "J2": "J2", "J3": "J3"}
