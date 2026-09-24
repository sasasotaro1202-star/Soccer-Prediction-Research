from src.data.fixture_field_audit import TARGET_COMPETITIONS, COMPETITION_NAMES, coverage_matrix
import pandas as pd


def test_target_competitions_match_canonical_name_catalog():
    assert len(TARGET_COMPETITIONS) == len(set(TARGET_COMPETITIONS))
    assert set(TARGET_COMPETITIONS) == set(COMPETITION_NAMES)


def test_unobserved_competitions_are_not_promoted_to_available():
    history = pd.DataFrame([
        {
            "competition": "EPL", "season": "2024/25", "source_name": "test",
            "home_goals": 1, "away_goals": 0,
        }
    ])
    matrix = coverage_matrix(history, pd.DataFrame())
    uecl = matrix[(matrix["competition"] == "UECL") & (matrix["source"] == "current_observed_adapter")]
    # The audit intentionally emits one explicit fixture-cell row per requested
    # season. Unobserved competitions therefore remain unavailable in every
    # requested season cell rather than being promoted to AVAILABLE.
    assert len(uecl) == 16
    assert set(uecl["status"]) == {"UNAVAILABLE"}
    assert all("not a claim" in str(reason) for reason in uecl["reason"])


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
