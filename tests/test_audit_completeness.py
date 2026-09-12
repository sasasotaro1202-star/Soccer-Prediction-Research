from src.data.fixture_field_audit import TARGET_COMPETITIONS, COMPETITION_NAMES, coverage_matrix
import pandas as pd


def test_target_competitions_are_exactly_fourteen():
    assert len(TARGET_COMPETITIONS) == 14
    assert len(set(TARGET_COMPETITIONS)) == 14
    assert set(TARGET_COMPETITIONS) == set(COMPETITION_NAMES)


def test_unobserved_competitions_are_not_promoted_to_available():
    history = pd.DataFrame([
        {
            "competition": "EPL", "season": "2024/25", "source_name": "test",
            "home_goals": 1, "away_goals": 0,
        }
    ])
    matrix = coverage_matrix(history, pd.DataFrame())
    ucl = matrix[(matrix["competition"] == "UCL") & (matrix["source"] == "current_observed_adapter")]
    assert len(ucl) == 1
    assert ucl.iloc[0]["status"] == "UNAVAILABLE"
    assert "not a claim" in ucl.iloc[0]["reason"]


def test_missing_is_not_real_zero():
    history = pd.DataFrame([
        {
            "competition": "EPL", "season": "2024/25", "source_name": "test",
            "home_goals": pd.NA, "away_goals": 0,
        }
    ])
    matrix = coverage_matrix(history, pd.DataFrame())
    row = matrix[(matrix["competition"] == "EPL") & (matrix["season"] == "2024/25") & (matrix["field"] == "home_goals")]
    assert row.iloc[0]["status"] == "UNAVAILABLE"
    assert row.iloc[0]["matched_count"] == 0
