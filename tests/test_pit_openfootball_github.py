import pandas as pd

from src.data.pit_openfootball_github import _row_key, _season


def test_openfootball_season_path_convention():
    assert _season(2024) == "2024-25"


def test_openfootball_row_identity_is_stable():
    row = pd.Series({
        "kickoff_utc": pd.Timestamp("2025-02-11T18:45:00Z"),
        "home_team": "Manchester City FC (ENG)",
        "away_team": "Real Madrid CF (ESP)",
        "home_goals": 2,
        "away_goals": 3,
    })
    assert _row_key(row) == (
        "2025-02-11",
        "Manchester City FC (ENG)",
        "Real Madrid CF (ESP)",
        2.0,
        3.0,
    )
