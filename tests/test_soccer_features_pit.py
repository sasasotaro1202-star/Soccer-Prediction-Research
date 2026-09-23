import pandas as pd

from src.features.soccer_features import build_match_features


def _game(match_id, kickoff, home, away, hg, ag, available):
    return {
        "match_id": match_id,
        "competition": "EPL",
        "season": "2025",
        "season_start": 2025,
        "kickoff_utc": pd.Timestamp(kickoff, tz="UTC"),
        "home_team": home,
        "away_team": away,
        "home_goals": hg,
        "away_goals": ag,
        "source_available_at_utc": pd.Timestamp(available, tz="UTC"),
        "neutral_venue": False,
    }


def test_future_match_never_enters_prior_state_even_with_bad_publication_time():
    history = pd.DataFrame(
        [
            _game("past", "2025-01-01 12:00", "A", "B", 1, 0, "2025-01-01 14:00"),
            # Deliberately malformed metadata: publication timestamp is before target,
            # but the match event itself is after target. Event-time ordering must win.
            _game("future", "2025-01-03 12:00", "A", "C", 9, 9, "2025-01-02 00:00"),
        ]
    )
    target = pd.DataFrame(
        [
            {
                "match_id": "target",
                "competition": "EPL",
                "season": "2025",
                "season_start": 2025,
                "kickoff_utc": pd.Timestamp("2025-01-02 12:00", tz="UTC"),
                "home_team": "A",
                "away_team": "B",
                "neutral_venue": False,
            }
        ]
    )
    row = build_match_features(history, target, windows=(1,)).iloc[0]
    assert row["home_gf_1"] == 1.0
    assert row["away_gf_1"] == 0.0
    assert 9.0 not in {row["home_gf_1"], row["away_gf_1"]}


def test_unknown_publication_time_does_not_become_pit_verified():
    history = pd.DataFrame(
        [
            _game("past", "2025-01-01 12:00", "A", "B", 1, 0, "2025-01-01 14:00"),
        ]
    )
    history.loc[0, "source_available_at_utc"] = pd.NaT
    target = pd.DataFrame(
        [
            {
                "match_id": "target",
                "competition": "EPL",
                "season": "2025",
                "season_start": 2025,
                "kickoff_utc": pd.Timestamp("2025-01-02 12:00", tz="UTC"),
                "home_team": "A",
                "away_team": "B",
                "neutral_venue": False,
            }
        ]
    )
    row = build_match_features(history, target, windows=(1,)).iloc[0]
    assert bool(row["pit_verified"]) is False


def test_pit_features_include_momentum_and_matchup_interactions():
    rows = []
    for i in range(12):
        home = "A" if i % 2 == 0 else "B"
        away = "B" if i % 2 == 0 else "A"
        rows.append(_game(
            f"m{i}",
            f"2025-01-{i+1:02d} 12:00",
            home,
            away,
            1 if i % 3 else 0,
            0 if i % 2 else 1,
            f"2025-01-{i+1:02d} 14:00",
        ))
    history = pd.DataFrame(rows)
    out = build_match_features(history, history)
    assert "points_ewma_momentum_diff_3v10" in out.columns
    assert "attack_defense_matchup_diff_5" in out.columns
    assert "draw_tension_10" in out.columns
    assert "strength_rest_interaction" in out.columns
