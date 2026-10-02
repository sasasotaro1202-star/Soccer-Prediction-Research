from src.research.row_pit_lineage import build_row_pit_lineage


def test_row_pit_lineage_passes_with_feature_available_before_cutoff():
    row = build_row_pit_lineage(
        match_id="m1",
        kickoff_utc="2026-01-10T18:00:00+00:00",
        prediction_cutoff_at_utc="2026-01-10T17:00:00+00:00",
        feature_source_max_available_at_utc="2026-01-10T16:30:00+00:00",
        pit_verified=True,
        outcome_source_available_at_utc="2026-01-10T21:00:00+00:00",
    )
    assert row["status"] == "PASS"
    assert row["pit_lineage_hash"].startswith("pit:")
    assert len(row["pit_lineage_hash"]) == 28


def test_row_pit_lineage_fails_closed_for_feature_after_cutoff():
    row = build_row_pit_lineage(
        match_id="m1",
        kickoff_utc="2026-01-10T18:00:00+00:00",
        prediction_cutoff_at_utc="2026-01-10T17:00:00+00:00",
        feature_source_max_available_at_utc="2026-01-10T17:00:01+00:00",
        pit_verified=True,
    )
    assert row["status"] == "FAIL_FEATURE_AFTER_CUTOFF"


def test_row_pit_lineage_fails_closed_when_cutoff_or_feature_timestamp_is_missing():
    missing_cutoff = build_row_pit_lineage(
        match_id="m1",
        kickoff_utc="2026-01-10T18:00:00+00:00",
        prediction_cutoff_at_utc=None,
        feature_source_max_available_at_utc="2026-01-10T16:30:00+00:00",
        pit_verified=True,
    )
    missing_feature = build_row_pit_lineage(
        match_id="m1",
        kickoff_utc="2026-01-10T18:00:00+00:00",
        prediction_cutoff_at_utc="2026-01-10T17:00:00+00:00",
        feature_source_max_available_at_utc=None,
        pit_verified=True,
    )
    assert missing_cutoff["status"] == "FAIL_MISSING_OR_INVALID_PREDICTION_CUTOFF"
    assert missing_feature["status"] == "FAIL_MISSING_OR_INVALID_FEATURE_AVAILABILITY"
