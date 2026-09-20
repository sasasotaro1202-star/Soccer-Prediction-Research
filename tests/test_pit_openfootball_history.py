from __future__ import annotations

import pandas as pd

from src.data.pit_openfootball_history import _lower_bound, _row_key, _season_path, _snapshot_keys


def test_openfootball_snapshot_parser_and_key():
    raw = '{"matches":[{"date":"2024-08-17","team1":"Arsenal","team2":"Wolves","score":{"ft":[2,0]}},{"date":"2024-08-18","team1":"Chelsea","team2":"Man City","score":[0,2]}]}'
    keys = _snapshot_keys(raw)
    assert ("2024-08-17", "arsenal", "wolves", 2, 0) in keys
    assert ("2024-08-18", "chelsea", "mancity", 0, 2) in keys


def test_openfootball_season_path():
    assert _season_path(2024, "en.1.json") == "2024-25/en.1.json"


def test_openfootball_row_key_and_conservative_bound():
    row = pd.Series({
        "kickoff_utc": "2024-08-17T15:00:00Z",
        "kickoff_time_available": True,
        "home_team": "Arsenal",
        "away_team": "Wolves",
        "home_goals": 2,
        "away_goals": 0,
    })
    assert _row_key(row) == ("2024-08-17", "arsenal", "wolves", 2, 0)
    assert _lower_bound(row).isoformat() == "2024-08-17T18:00:00+00:00"
