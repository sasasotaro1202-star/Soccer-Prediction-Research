import numpy as np
import pandas as pd

from src.features.soccer_features import ELO_HOME_ADV, _update_elo, build_match_features


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


def test_neutral_elo_does_not_apply_home_advantage():
    normal = {"global": {}, "competition": {}}
    neutral = {"global": {}, "competition": {}}
    _update_elo(normal, "A", "B", 0, "AG_M", neutral_venue=False)
    _update_elo(neutral, "A", "B", 0, "AG_M", neutral_venue=True)

    # With equal starting ratings, a normal venue expects A to win above 50%;
    # a neutral venue starts at exactly 50%.
    expected_normal = 1.0 / (1.0 + 10.0 ** (-ELO_HOME_ADV / 400.0))
    delta_normal = 20.0 * (1.0 - expected_normal)
    delta_neutral = 20.0 * (1.0 - 0.5)
    np.testing.assert_allclose(
        normal["global"]["A"], 1500.0 + delta_normal, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        neutral["global"]["A"], 1500.0 + delta_neutral, rtol=0, atol=1e-12
    )
    assert neutral["global"]["A"] < normal["global"]["A"]


def test_build_features_respects_neutral_venue_and_pit():
    history = pd.DataFrame([
        _row("h1", "2026-09-01 12:00", "A", "B", 1, 0, "2026-09-01 10:00", neutral=True),
        _row("h2", "2026-09-02 12:00", "B", "A", 0, 1, "2026-09-02 10:00", neutral=True),
    ])
    matches = pd.DataFrame([
        {
            "match_id": "target",
            "competition": "AG_M",
            "season": "2026",
            "season_start": 2026,
            "kickoff_utc": pd.Timestamp("2026-09-03 12:00", tz="UTC"),
            "home_team": "A",
            "away_team": "B",
            "neutral_venue": True,
        }
    ])

    out = build_match_features(history, matches, windows=(1,))
    row = out.iloc[0]
    assert bool(row["neutral_venue"]) is True
    assert bool(row["neutral_venue_known"]) is True
    assert row["home_advantage"] == 0.0
    assert abs(row["home_elo_expected"] - 0.5) < 1e-12
    assert bool(row["pit_verified"]) is True


def test_unknown_neutral_venue_fails_closed():
    history = pd.DataFrame([
        _row("h1", "2026-09-01 12:00", "A", "B", 1, 0, "2026-09-01 10:00", neutral=True),
        _row("h2", "2026-09-02 12:00", "B", "A", 0, 1, "2026-09-02 10:00", neutral=True),
    ])
    matches = pd.DataFrame([
        {
            "match_id": "target",
            "competition": "AG_M",
            "season": "2026",
            "season_start": 2026,
            "kickoff_utc": pd.Timestamp("2026-09-03 12:00", tz="UTC"),
            "home_team": "A",
            "away_team": "B",
            "neutral_venue": pd.NA,
        }
    ])

    out = build_match_features(history, matches, windows=(1,))
    row = out.iloc[0]
    assert bool(row["neutral_venue_known"]) is False
    assert pd.isna(row["home_advantage"])
    assert pd.isna(row["home_elo_expected"])
    assert bool(row["pit_verified"]) is False
