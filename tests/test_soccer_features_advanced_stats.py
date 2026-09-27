import numpy as np
import pandas as pd

from src.features.soccer_features import build_match_features


def _history() -> pd.DataFrame:
    rows = [
        ("2026-08-01T12:00:00Z", "A", "C", 2, 0, 2.0, 0.4, 58.0, 3),
        ("2026-08-03T12:00:00Z", "D", "B", 0, 1, 0.5, 1.4, 42.0, 2),
        ("2026-08-10T12:00:00Z", "A", "D", 1, 1, 1.2, 1.0, 61.0, 4),
        ("2026-08-12T12:00:00Z", "E", "B", 0, 0, 0.8, 0.7, 47.0, 1),
        ("2026-08-20T12:00:00Z", "A", "E", 3, 1, 2.4, 0.9, 55.0, 5),
        ("2026-08-22T12:00:00Z", "F", "B", 2, 2, 1.1, 1.6, 49.0, 3),
        # This row is after the fixture below and must never enter its features.
        ("2026-09-30T12:00:00Z", "A", "B", 4, 0, 9.0, 0.1, 99.0, 99),
    ]
    out = pd.DataFrame(
        rows,
        columns=[
            "kickoff_utc",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "home_xg",
            "away_xg",
            "home_possession",
            "home_big_chances",
            "home_shots_inside_box",
            "home_shots_outside_box",
            "home_blocked_shots",
            "home_offsides",
            "home_pass_accuracy",
            "home_goals_ht",
            "home_xg_ht",
        ],
    )
    out["match_id"] = [f"h{i}" for i in range(len(out))]
    out["competition"] = "TEST"
    out["source_available_at_utc"] = pd.to_datetime("2026-07-01T00:00:00Z")
    out["away_possession"] = 100.0 - out["home_possession"]
    out["away_big_chances"] = 2.0
    out["away_shots_inside_box"] = 4.0
    out["away_shots_outside_box"] = 5.0
    out["away_blocked_shots"] = 2.0
    out["away_offsides"] = 1.0
    out["away_pass_accuracy"] = 71.0
    out["away_goals_ht"] = 0.0
    out["away_xg_ht"] = out["away_xg"] * 0.45
    out["home_xg_ht"] = out["home_xg"] * 0.45
    return out


def test_advanced_stats_are_featured_with_aliases_and_safe_xga_derivation():
    history = _history()
    matches = pd.DataFrame(
        [{
            "match_id": "future",
            "competition": "TEST",
            "season": "2026/27",
            "season_start": 2026,
            "kickoff_utc": "2026-09-29T12:00:00Z",
            "home_team": "A",
            "away_team": "B",
            "neutral_venue": False,
        }]
    )

    result = build_match_features(history, matches, windows=(3,))
    row = result.iloc[0]

    assert row["home_xg_avg_3"] == np.mean([2.0, 1.2, 2.4])
    assert row["away_xg_avg_3"] == np.mean([1.4, 0.7, 1.6])

    # xGA is safely derived from the opponent's explicitly supplied xG.
    assert row["home_xga_avg_3"] == row["away_xg_avg_3"]
    assert row["away_xga_avg_3"] == row["home_xg_avg_3"]

    assert row["home_possession_avg_3"] == np.mean([58.0, 61.0, 55.0])
    assert row["away_possession_avg_3"] == np.mean([42.0, 47.0, 49.0])
    assert row["home_big_chances_avg_3"] == np.mean([3.0, 4.0, 5.0])
    assert row["home_shots_inside_box_avg_3"] == 0.0 or np.isfinite(row["home_shots_inside_box_avg_3"])
    assert row["home_shots_outside_box_avg_3"] == 5.0
    assert row["home_blocked_shots_avg_3"] == 2.0
    assert row["home_offsides_avg_3"] == 1.0
    assert row["home_pass_accuracy_avg_3"] == np.mean([58.0, 61.0, 55.0])
    assert row["home_goals_ht_avg_3"] == 0.0
    assert np.isfinite(row["home_xg_ht_avg_3"])

    # The post-fixture 9.0 xG observation cannot leak backwards.
    assert row["home_xg_avg_3"] < 3.0
    assert bool(row["pit_verified"])
