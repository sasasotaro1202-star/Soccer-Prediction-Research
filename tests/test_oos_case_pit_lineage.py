from __future__ import annotations

import pytest

from src.evaluation.walk_forward import _pit_lineage_for_case


def _row(**overrides):
    row = {
        "match_id": "m1",
        "kickoff_utc": "2026-10-01T18:00:00Z",
        "prediction_cutoff_at_utc": "2026-10-01T17:00:00Z",
        "feature_source_max_available_at_utc": "2026-10-01T16:30:00Z",
        "source_available_at_utc": "2026-10-01T20:15:00Z",
        "pit_verified": True,
    }
    row.update(overrides)
    return row


def test_pit_lineage_separates_prediction_features_from_outcome_publication():
    result = _pit_lineage_for_case(_row())
    assert result["pit_lineage_status"] == "PASS"
    assert result["prediction_cutoff_at_utc"].startswith("2026-10-01T17:00:00")
    assert result["feature_source_max_available_at_utc"].startswith("2026-10-01T16:30:00")
    assert result["outcome_source_available_at_utc"].startswith("2026-10-01T20:15:00")
    assert str(result["pit_lineage_hash"]).startswith("pit:")
    assert _pit_lineage_for_case(_row())["pit_lineage_hash"] == result["pit_lineage_hash"]


def test_pit_lineage_fails_closed_on_missing_prediction_cutoff():
    with pytest.raises(RuntimeError, match="missing prediction cutoff"):
        _pit_lineage_for_case(_row(prediction_cutoff_at_utc=None))


def test_pit_lineage_fails_closed_on_late_feature_source():
    with pytest.raises(RuntimeError, match="PIT violation"):
        _pit_lineage_for_case(_row(feature_source_max_available_at_utc="2026-10-01T17:01:00Z"))


def test_pit_lineage_fails_closed_when_row_is_not_pit_verified():
    with pytest.raises(RuntimeError, match="pit_verified=True"):
        _pit_lineage_for_case(_row(pit_verified=False))
