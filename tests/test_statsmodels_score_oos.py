import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from src.research.external_oss.statsmodels_score_oos import run_statsmodels_score_oos


def _history(n=48):
    fixtures = [
        ("A", "B", 1, 0),
        ("B", "C", 0, 1),
        ("C", "A", 2, 1),
        ("A", "C", 0, 0),
        ("C", "B", 1, 2),
        ("B", "A", 2, 0),
    ]
    rows = []
    for i in range(n):
        home, away, hg, ag = fixtures[i % len(fixtures)]
        rows.append(
            {
                "match_id": f"oos-{i}",
                "kickoff_utc": f"2025-01-{(i % 28) + 1:02d}T12:00:00Z",
                "home_team": home,
                "away_team": away,
                "competition": "TEST",
                "home_goals": hg,
                "away_goals": ag,
                "pit_verified": True,
                "source_available_at_utc": f"2025-01-{(i % 28) + 1:02d}T18:00:00Z",
            }
        )
    # Use a monotonic sequence because the compact dates above repeat months.
    frame = pd.DataFrame(rows)
    frame["kickoff_utc"] = pd.date_range(
        "2025-01-01T12:00:00Z", periods=n, freq="2D"
    )
    frame["source_available_at_utc"] = frame["kickoff_utc"] - pd.Timedelta(hours=2)
    return frame


def test_chronological_oos_runner_is_selection_free_and_production_blocked():
    result = run_statsmodels_score_oos(
        _history(), min_train=24, oos_block=8, cutoff_buffer_minutes=1,
        calibration_min_rows=10,
        tests_passed=True, audit_passed=True
    )
    assert result["status"] == "RESEARCH_OOS_READY"
    assert result["selection_performed"] is False
    assert result["frozen_holdout_used"] is False
    assert result["production_usable"] is False
    assert result["fold_count"] == 3
    for row in result["rows"]:
        assert row["training_rows"] >= 24
        assert pd.Timestamp(row["prediction_cutoff_utc"]) > pd.Timestamp(row["oos_start_utc"]) - pd.Timedelta(minutes=2)


def test_unknown_source_timestamp_fails_closed():
    history = _history()
    history.loc[0, "source_available_at_utc"] = None
    with pytest.raises(ValueError, match="availability"):
        run_statsmodels_score_oos(history, min_train=24, oos_block=8, cutoff_buffer_minutes=1, calibration_min_rows=10, tests_passed=True, audit_passed=True)


def test_feature_replay_availability_is_used_when_legacy_source_time_is_missing():
    history = _history()
    history["feature_source_max_available_at_utc"] = history["kickoff_utc"] - pd.Timedelta(minutes=30)
    history["source_available_at_utc"] = None
    result = run_statsmodels_score_oos(
        history, min_train=24, oos_block=8, cutoff_buffer_minutes=1,
        calibration_min_rows=10,
        tests_passed=True, audit_passed=True
    )
    assert result["status"] == "RESEARCH_OOS_READY"
    assert result["fold_count"] == 3


def test_pit_availability_filter_does_not_lower_training_requirement():
    history = _history()
    history.loc[0, "source_available_at_utc"] = pd.Timestamp(
        "2030-01-01T00:00:00Z"
    )
    result = run_statsmodels_score_oos(
        history, min_train=24, oos_block=8, cutoff_buffer_minutes=1
    )
    assert result["status"] == "RESEARCH_OOS_READY"
    assert result["rows"][0]["training_rows"] >= 24
    assert all(row["pit_training_boundary_valid"] for row in result["rows"])
    assert all(row["same_kickoff_split_avoided"] for row in result["rows"])


def test_unsupported_oos_team_is_explicitly_excluded_not_imputed():
    history = _history(60)
    history.loc[50, "home_team"] = "UNSEEN"
    result = run_statsmodels_score_oos(
        history, min_train=24, oos_block=12, cutoff_buffer_minutes=1,
        calibration_min_rows=10,
        tests_passed=True, audit_passed=True
    )
    assert result["status"] == "RESEARCH_OOS_READY"
    assert sum(row["unsupported_rows"] for row in result["rows"]) >= 1
    assert all(row["common_evaluable_rows"] + row["unsupported_rows"] == row["oos_rows"] for row in result["rows"])


def test_missing_gate_handoff_is_blocked():
    result = run_statsmodels_score_oos(
        _history(), min_train=24, oos_block=8, cutoff_buffer_minutes=1,
        tests_passed=True, audit_passed=False
    )
    assert result["status"] == "BLOCKED"
    assert result["production_usable"] is False


def test_case_level_oos_pit_is_enforced():
    history = _history()
    history.loc[30, "source_available_at_utc"] = history.loc[30, "kickoff_utc"] + pd.Timedelta(hours=2)
    with pytest.raises(ValueError, match="case-level predictor PIT"):
        run_statsmodels_score_oos(
            history, min_train=24, oos_block=8, cutoff_buffer_minutes=1,
            tests_passed=True, audit_passed=True
        )
