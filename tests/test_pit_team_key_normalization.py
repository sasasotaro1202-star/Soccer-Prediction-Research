import pandas as pd

from src.data.pit_source_adapter_fast import _normalized_row_key, normalize_team_identity


def test_normalize_team_identity_removes_presentation_only_differences():
    assert normalize_team_identity("West Brom") == normalize_team_identity(" West-Brom ")
    assert normalize_team_identity("Atlético Madrid") == normalize_team_identity("Atletico Madrid")


def test_normalized_row_key_preserves_event_identity_fields():
    source = pd.Series(
        {
            "source_event_date": "2024-05-19",
            "home_team": "West Brom",
            "away_team": "Preston",
            "home_goals": 3,
            "away_goals": 0,
            "result": "H",
        }
    )
    archived = pd.Series(
        {
            "source_event_date": "2024-05-19",
            "home_team": "West-Brom",
            "away_team": "Preston ",
            "home_goals": 3.0,
            "away_goals": 0.0,
            "result": "H",
        }
    )
    assert _normalized_row_key(source) == _normalized_row_key(archived)


def test_normalized_row_key_does_not_ignore_goals_or_result():
    source = pd.Series(
        {
            "source_event_date": "2024-05-19",
            "home_team": "West Brom",
            "away_team": "Preston",
            "home_goals": 3,
            "away_goals": 0,
            "result": "H",
        }
    )
    changed_result = source.copy()
    changed_result["result"] = "D"
    changed_score = source.copy()
    changed_score["home_goals"] = 2
    assert _normalized_row_key(source) != _normalized_row_key(changed_result)
    assert _normalized_row_key(source) != _normalized_row_key(changed_score)
