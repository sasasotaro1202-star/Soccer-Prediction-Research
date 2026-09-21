import pandas as pd

from src.features.soccer_features import build_match_features


def _row(match_id, kickoff, home, away, hg, ag, available, neutral=True):
    return {
        "match_id": match_id,
        "competition": "AG_M",
        "season": "2026",
        "season_start": 2026,
        "kickoff_utc": pd.Timestamp(kickoff, tz="UTC"),
        "home_team": home,
        "away_team": away,
        "home_goals": hg,
        "away_goals": ag,
        "source_available_at_utc": pd.Timestamp(available, tz="UTC"),
        "neutral_venue": neutral,
    }


def test_neutral_venue_has_zero_home_advantage():
    history = pd.DataFrame([
        _row("h1", "2026-09-01 12:00", "A", "B", 1, 0, "2026-09-01 10:00", True),
        _row("h2", "2026-09-02 12:00", "B", "A", 0, 1, "2026-09-02 10:00", True),
    ])
    matches = pd.DataFrame([{
        "match_id": "target",
        "competition": "AG_M",
        "season": "2026",
        "season_start": 2026,
        "kickoff_utc": pd.Timestamp("2026-09-03 12:00", tz="UTC"),
        "home_team": "A",
        "away_team": "B",
        "neutral_venue": True,
    }])
    row = build_match_features(history, matches, windows=(1,)).iloc[0]
    assert bool(row["neutral_venue"]) is True
    assert bool(row["neutral_venue_known"]) is True
    assert row["home_advantage"] == 0.0
    assert bool(row["pit_verified"]) is True


def test_unknown_asian_games_venue_fails_closed():
    history = pd.DataFrame([
        _row("h1", "2026-09-01 12:00", "A", "B", 1, 0, "2026-09-01 10:00", True),
        _row("h2", "2026-09-02 12:00", "B", "A", 0, 1, "2026-09-02 10:00", True),
    ])
    matches = pd.DataFrame([{
        "match_id": "target",
        "competition": "AG_M",
        "season": "2026",
        "season_start": 2026,
        "kickoff_utc": pd.Timestamp("2026-09-03 12:00", tz="UTC"),
        "home_team": "A",
        "away_team": "B",
        "neutral_venue": pd.NA,
    }])
    row = build_match_features(history, matches, windows=(1,)).iloc[0]
    assert bool(row["neutral_venue_known"]) is False
    assert pd.isna(row["home_advantage"])
    assert pd.isna(row["home_elo_expected"])
    assert bool(row["pit_verified"]) is False
