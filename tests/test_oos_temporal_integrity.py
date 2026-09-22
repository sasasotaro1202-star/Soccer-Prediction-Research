import pandas as pd

from src.research.engine import _oos_temporal_integrity


def _frame(starts, ends):
    return pd.DataFrame({"oos_start": starts, "oos_end": ends})


def test_temporal_oos_integrity_accepts_strict_non_overlapping_blocks():
    frame = _frame(
        ["2020-01-01", "2020-06-01", "2021-01-01"],
        ["2020-05-31", "2020-12-31", "2021-06-30"],
    )
    result = _oos_temporal_integrity(frame, locked_blocks=2)
    assert result["status"] == "PASS"
    assert result["chronological"] is True
    assert result["non_overlapping"] is True
    assert result["development_before_locked"] is True


def test_temporal_oos_integrity_rejects_overlap():
    frame = _frame(
        ["2020-01-01", "2020-06-01", "2020-12-01"],
        ["2020-07-01", "2020-12-31", "2021-03-31"],
    )
    result = _oos_temporal_integrity(frame, locked_blocks=2)
    assert result["status"] == "FAIL"
    assert result["non_overlapping"] is False
    assert result["reason"] == "oos_blocks_overlap"


def test_temporal_oos_integrity_rejects_unsorted_blocks():
    frame = _frame(
        ["2020-06-01", "2020-01-01", "2021-01-01"],
        ["2020-12-31", "2020-05-31", "2021-06-30"],
    )
    result = _oos_temporal_integrity(frame, locked_blocks=2)
    assert result["status"] == "FAIL"
    assert result["chronological"] is False
    assert result["reason"] == "oos_blocks_not_strictly_chronological"


def test_temporal_oos_integrity_fails_closed_on_missing_boundaries():
    frame = pd.DataFrame({"oos_start": ["2020-01-01", "2020-06-01", "2021-01-01"]})
    result = _oos_temporal_integrity(frame, locked_blocks=2)
    assert result["status"] == "FAIL"
    assert result["has_boundaries"] is False


def test_temporal_oos_integrity_requires_development_and_locked_blocks():
    frame = _frame(["2020-01-01", "2020-06-01"], ["2020-05-31", "2020-12-31"])
    result = _oos_temporal_integrity(frame, locked_blocks=2)
    assert result["status"] == "FAIL"
    assert result["reason"] == "insufficient_blocks_for_development_and_locked_holdout"
