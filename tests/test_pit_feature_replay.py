import pandas as pd

from src.features.soccer_features import build_match_features


def _history():
    return pd.DataFrame([
        {
            "match_id": "m1", "competition": "EPL", "season": "2024/25",
            "kickoff_utc": pd.Timestamp("2025-01-01", tz="UTC"),
            "home_team": "A", "away_team": "B", "home_goals": 2, "away_goals": 0,
            "source_available_at_utc": pd.Timestamp("2025-01-01 20:00", tz="UTC"),
        },
        {
            "match_id": "m2", "competition": "EPL", "season": "2024/25",
            "kickoff_utc": pd.Timestamp("2025-01-03", tz="UTC"),
            "home_team": "A", "away_team": "C", "home_goals": 1, "away_goals": 0,
            "source_available_at_utc": pd.Timestamp("2025-01-03 20:00", tz="UTC"),
        },
        {
            "match_id": "m3", "competition": "EPL", "season": "2024/25",
            "kickoff_utc": pd.Timestamp("2025-01-05", tz="UTC"),
            "home_team": "D", "away_team": "A", "home_goals": 0, "away_goals": 1,
            "source_available_at_utc": pd.Timestamp("2025-01-05 20:00", tz="UTC"),
        },
    ])


def test_unavailable_recent_result_invalidates_window():
    history = _history()
    match = pd.DataFrame([{
        "match_id": "m4", "competition": "EPL", "season": "2024/25",
        "kickoff_utc": pd.Timestamp("2025-01-06", tz="UTC"),
        "home_team": "A", "away_team": "E",
    }])
    features = build_match_features(history, match, windows=(3,))
    assert features.iloc[0]["pit_verified"] is False
    assert pd.isna(features.iloc[0]["home_gf_3"])


def test_all_required_history_available_makes_window_pit_valid():
    history = _history()
    history["source_available_at_utc"] = pd.to_datetime("2024-12-31", utc=True)
    match = pd.DataFrame([{
        "match_id": "m4", "competition": "EPL", "season": "2024/25",
        "kickoff_utc": pd.Timestamp("2025-01-06", tz="UTC"),
        "home_team": "A", "away_team": "E",
    }])
    features = build_match_features(history, match, windows=(3,))
    assert bool(features.iloc[0]["pit_verified"])
    assert features.iloc[0]["home_gf_3"] == 4 / 3
