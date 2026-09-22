import pandas as pd

from src.features.soccer_features import add_target, build_match_features


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


def test_unavailable_recent_result_is_excluded_from_sparse_pit_window():
    history = _history()
    history.loc[5, "source_available_at_utc"] = pd.Timestamp("2025-01-08 01:00", tz="UTC")
    match = pd.DataFrame([{
        "match_id": "m7", "competition": "EPL", "season": "2024/25",
        "kickoff_utc": pd.Timestamp("2025-01-08", tz="UTC"),
        "home_team": "A", "away_team": "B",
    }])
    features = build_match_features(history, match, windows=(3,))
    # m6 is not available by the cutoff, so it is excluded. The model may still
    # use the last three explicitly available matches; unknown timing is never
    # treated as safe.
    assert bool(features.iloc[0]["pit_verified"])
    assert pd.notna(features.iloc[0]["home_gf_3"])


def test_delayed_result_is_available_at_later_cutoff():
    history = _history()
    history.loc[5, "source_available_at_utc"] = pd.Timestamp("2025-01-08 01:00", tz="UTC")
    matches = pd.DataFrame([
        {
            "match_id": "m7", "competition": "EPL", "season": "2024/25",
            "kickoff_utc": pd.Timestamp("2025-01-08", tz="UTC"),
            "home_team": "A", "away_team": "B",
        },
        {
            "match_id": "m8", "competition": "EPL", "season": "2024/25",
            "kickoff_utc": pd.Timestamp("2025-01-09", tz="UTC"),
            "home_team": "A", "away_team": "B",
        },
    ])
    features = build_match_features(history, matches, windows=(3,))
    assert pd.notna(features.iloc[1]["home_gf_3"])
    assert bool(features.iloc[1]["pit_verified"])


def test_delayed_row_does_not_hide_later_available_history():
    history = _history()
    history.loc[4, "source_available_at_utc"] = pd.Timestamp("2025-01-06 18:00", tz="UTC")
    history.loc[5, "source_available_at_utc"] = pd.Timestamp("2025-01-10 01:00", tz="UTC")
    matches = pd.DataFrame([
        {
            "match_id": "m7", "competition": "EPL", "season": "2024/25",
            "kickoff_utc": pd.Timestamp("2025-01-07", tz="UTC"),
            "home_team": "A", "away_team": "B",
        },
        {
            "match_id": "m8", "competition": "EPL", "season": "2024/25",
            "kickoff_utc": pd.Timestamp("2025-01-08", tz="UTC"),
            "home_team": "A", "away_team": "B",
        },
    ])
    features = build_match_features(history, matches, windows=(3,))
    # m6 is unknown at m7/m8 cutoffs, but m5 is known and must not be
    # skipped merely because the later event is still unavailable.
    assert features.iloc[0]["home_gf_3"] == 5 / 3
    assert features.iloc[1]["home_gf_3"] == 5 / 3
    assert bool(features.iloc[0]["pit_verified"])
    assert bool(features.iloc[1]["pit_verified"])


def test_unknown_publication_time_is_not_silently_inferred():
    history = _history()
    history["source_available_at_utc"] = pd.NaT
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
    assert features.iloc[0]["home_gf_3"] == 4 / 3


def test_two_explicitly_available_history_rows_are_pit_valid(monkeypatch):
    monkeypatch.setenv("SOCCER_MIN_PIT_HISTORY_GAMES", "2")
    history = _history().iloc[:4].copy()
    match = pd.DataFrame([{
        "match_id": "m5", "competition": "EPL", "season": "2024/25",
        "kickoff_utc": pd.Timestamp("2025-01-07", tz="UTC"),
        "home_team": "A", "away_team": "B",
    }])
    features = build_match_features(history, match, windows=(3, 5))
    assert bool(features.iloc[0]["pit_verified"])
    assert pd.notna(features.iloc[0]["home_gf_3"])


def test_missing_match_outcome_does_not_become_away_win():
    features = pd.DataFrame([
        {"match_id": "m1", "kickoff_utc": pd.Timestamp("2025-01-01", tz="UTC")},
        {"match_id": "m2", "kickoff_utc": pd.Timestamp("2025-01-02", tz="UTC")},
    ])
    matches = pd.DataFrame([
        {"match_id": "m1", "home_goals": 2, "away_goals": 1},
        {"match_id": "m2", "home_goals": pd.NA, "away_goals": pd.NA},
    ])
    out = add_target(features, matches)
    assert out.loc[0, "target"] == 0
    assert pd.isna(out.loc[1, "target"])
