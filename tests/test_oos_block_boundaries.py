import pandas as pd

from src.evaluation.walk_forward import _advance_past_same_kickoff as advance_1x2
from src.evaluation.score_walk_forward import _advance_past_same_kickoff as advance_score


def _frame():
    return pd.DataFrame(
        {
            "kickoff_utc": pd.to_datetime(
                [
                    "2024-01-01T12:00:00Z",
                    "2024-01-01T12:00:00Z",
                    "2024-01-01T12:00:00Z",
                    "2024-01-02T12:00:00Z",
                ],
                utc=True,
            )
        }
    )


def test_1x2_boundary_moves_past_identical_kickoff_group():
    frame = _frame()
    assert advance_1x2(frame, 1) == 3
    assert advance_1x2(frame, 3) == 3
    assert advance_1x2(frame, 4) == 4


def test_score_boundary_moves_past_identical_kickoff_group():
    frame = _frame()
    assert advance_score(frame, 1) == 3
    assert advance_score(frame, 3) == 3
    assert advance_score(frame, 4) == 4
