import pandas as pd

from src.data.fixture_field_audit import TARGET_COMPETITIONS, coverage_matrix, field_audit, fixture_audit, source_reconciliation


def sample_history():
    return pd.DataFrame([
        {
            "match_id": "fd:EPL:2025:1",
            "competition": "EPL",
            "season": "2025/26",
            "home_team": "A",
            "away_team": "B",
            "kickoff_utc": pd.Timestamp("2025-08-01T19:00:00Z"),
            "event_time_precision": "MINUTE",
            "result": "H",
            "home_goals": 0,
            "away_goals": 2,
            "home_shots": 0,
            "away_shots": 10,
            "home_shots_on_target": 0,
            "away_shots_on_target": 4,
            "home_corners": 0,
            "away_corners": 5,
            "home_fouls": 7,
            "away_fouls": 8,
            "home_yellow_cards": 0,
            "away_yellow_cards": 1,
            "home_red_cards": 0,
            "away_red_cards": 0,
            "source_name": "Football-Data.co.uk",
            "source_record_id": "1",
            "source_available_at_utc": pd.NaT,
            "retrieved_at_utc": pd.Timestamp("2026-01-01T00:00:00Z"),
        }
    ])


def test_target_scope_has_unique_canonical_competitions():
    assert len(TARGET_COMPETITIONS) == len(set(TARGET_COMPETITIONS))


def test_fixture_audit_keeps_canonical_identity():
    out = fixture_audit(sample_history())
    assert len(out) == 1
    assert out.iloc[0]["source"] == "Football-Data.co.uk"
    assert out.iloc[0]["source_fixture_id"] == "1"


def test_field_audit_distinguishes_real_zero_from_missing_and_pit_unknown():
    out = field_audit(sample_history())
    zero = out[out.field_name == "home_shots"].iloc[0]
    assert zero.value_status == "REAL_ZERO"
    assert zero.pit_status == "PIT_UNKNOWN"

    h = sample_history()
    h.loc[0, "home_corners"] = pd.NA
    out2 = field_audit(h)
    missing = out2[out2.field_name == "home_corners"].iloc[0]
    assert missing.value_status == "MISSING"


def test_own_match_result_is_not_pit_safe():
    out = field_audit(sample_history())
    result = out[out.field_name == "result"].iloc[0]
    assert result.pit_status == "PIT_UNSAFE"


def test_coverage_marks_unobserved_competitions_unavailable_not_zero():
    h = sample_history()
    f = field_audit(h)
    c = coverage_matrix(h, f)
    row = c[c.competition == "UECL"].iloc[0]
    assert row.status == "UNAVAILABLE"
    assert row.fixture_count == 0
    reason = row.reason.lower()
    assert "not a claim" in reason
    assert "data exists elsewhere" in reason


def test_reconciliation_flags_duplicate_rows_without_double_counting_source():
    h = pd.concat([sample_history(), sample_history()], ignore_index=True)
    out = source_reconciliation(h)
    assert out.iloc[0].source_count == 1
    assert out.iloc[0].duplicate_source_identity is True
