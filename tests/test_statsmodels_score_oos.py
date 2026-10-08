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
    frame["source_available_at_utc"] = frame["kickoff_utc"] + pd.Timedelta(hours=6)
    return frame


def test_chronological_oos_runner_is_selection_free_and_production_blocked():
    result = run_statsmodels_score_oos(
        _history(), min_train=24, oos_block=8, cutoff_buffer_minutes=1
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
    with pytest.raises(ValueError, match="unknown timestamps"):
        run_statsmodels_score_oos(history, min_train=24, oos_block=8, cutoff_buffer_minutes=1)


def test_feature_replay_availability_is_used_when_legacy_source_time_is_missing():
    history = _history()
    history["feature_source_max_available_at_utc"] = history["kickoff_utc"] - pd.Timedelta(minutes=30)
    history["source_available_at_utc"] = None
    result = run_statsmodels_score_oos(
        history, min_train=24, oos_block=8, cutoff_buffer_minutes=1
    )
    assert result["status"] == "RESEARCH_OOS_READY"
    assert result["fold_count"] == 3
