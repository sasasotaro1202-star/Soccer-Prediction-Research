import pandas as pd

from src.features.soccer_features import build_match_features


def _history():
    rows = []
    dates = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06"]
    for i, date in enumerate(dates):
        rows.append({
            "match_id": f"m{i+1}", "competition": "EPL", "season": "2024/25",
            "kickoff_utc": pd.Timestamp(date, tz="UTC"),
            "home_team": "A" if i % 2 == 0 else "B",
            "away_team": "B" if i % 2 == 0 else "A",
            "home_goals": 2 if i % 2 == 0 else 0,
            "away_goals": 0 if i % 2 == 0 else 1,
            "source_available_at_utc": pd.Timestamp(f"{date} 20:00", tz="UTC"),
        })
    return pd.DataFrame(rows)


def test_unavailable_recent_result_invalidates_window():
    history = _history()
    # m6 becomes available after the prediction cutoff (2025-01-07 23:00 UTC).
    history.loc[5, "source_available_at_utc"] = pd.Timestamp("2025-01-08 01:00", tz="UTC")
    match = pd.DataFrame([{
        "match_id": "m7", "competition": "EPL", "season": "2024/25",
        "kickoff_utc": pd.Timestamp("2025-01-08", tz="UTC"),
        "home_team": "A", "away_team": "B",
    }])
    features = build_match_features(history, match, windows=(3,))
    assert not bool(features.iloc[0]["pit_verified"])
    assert pd.isna(features.iloc[0]["home_gf_3"])


def test_all_required_history_available_makes_window_pit_valid():
    history = _history()
    history["source_available_at_utc"] = pd.to_datetime("2024-12-31", utc=True)
    match = pd.DataFrame([{
        "match_id": "m7", "competition": "EPL", "season": "2024/25",
        "kickoff_utc": pd.Timestamp("2025-01-08", tz="UTC"),
        "home_team": "A", "away_team": "B",
    }])
    features = build_match_features(history, match, windows=(3,))
    assert bool(features.iloc[0]["pit_verified"])
    assert features.iloc[0]["home_gf_3"] == 5 / 3
