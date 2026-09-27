import pandas as pd

from src.features.soccer_features import build_match_features


def test_advanced_stats_are_replayed_into_pit_safe_rolling_features():
    history_rows = []
    teams = [("Alpha", "Beta"), ("Gamma", "Delta"), ("Alpha", "Beta")]
    dates = ["2026-08-20T12:00:00Z", "2026-08-25T12:00:00Z", "2026-09-01T12:00:00Z"]
    for i, (home, away) in enumerate(teams):
        history_rows.append({
            "match_id": f"h{i}",
            "competition": "EPL",
            "kickoff_utc": dates[i],
            "home_team": home,
            "away_team": away,
            "home_goals": 2,
            "away_goals": 1,
            "source_available_at_utc": f"2026-08-{21+i*5:02d}T12:00:00Z",
            "home_xg": 1.0,
            "away_xg": 0.5,
            "home_possession": 60.0,
            "away_possession": 40.0,
            "home_pass_accuracy": 82.0,
            "away_pass_accuracy": 77.0,
            "home_shots_inside_box": 8.0,
            "away_shots_inside_box": 4.0,
            "home_shots_outside_box": 5.0,
            "away_shots_outside_box": 6.0,
            "home_blocked_shots": 2.0,
            "away_blocked_shots": 3.0,
            "home_penalties": 0.0,
            "away_penalties": 1.0,
            "home_offsides": 2.0,
            "away_offsides": 1.0,
            "home_goals_ht": 1.0,
            "away_goals_ht": 0.0,
            "home_xg_ht": 0.4,
            "away_xg_ht": 0.2,
        })
    # Add two additional Alpha/Beta matches so the required PIT window is fully populated.
    for j, day in enumerate((8, 15), start=3):
        history_rows.append({
            "match_id": f"h{j}",
            "competition": "EPL",
            "kickoff_utc": f"2026-09-{day:02d}T12:00:00Z",
            "home_team": "Alpha",
            "away_team": "Beta",
            "home_goals": 1,
            "away_goals": 1,
            "source_available_at_utc": f"2026-09-{day+1:02d}T12:00:00Z",
            "home_xg": 1.0,
            "away_xg": 0.5,
            "home_possession": 60.0,
            "away_possession": 40.0,
            "home_pass_accuracy": 82.0,
            "away_pass_accuracy": 77.0,
            "home_shots_inside_box": 8.0,
            "away_shots_inside_box": 4.0,
            "home_shots_outside_box": 5.0,
            "away_shots_outside_box": 6.0,
            "home_blocked_shots": 2.0,
            "away_blocked_shots": 3.0,
            "home_penalties": 0.0,
            "away_penalties": 1.0,
            "home_offsides": 2.0,
            "away_offsides": 1.0,
            "home_goals_ht": 1.0,
            "away_goals_ht": 0.0,
            "home_xg_ht": 0.4,
            "away_xg_ht": 0.2,
        })

    # This match happened before the prediction cutoff but its advanced statistics
    # were published after the cutoff. It must not enter the rolling feature state.
    history_rows.append({
        "match_id": "late_known",
        "competition": "EPL",
        "kickoff_utc": "2026-09-18T12:00:00Z",
        "home_team": "Alpha",
        "away_team": "Beta",
        "home_goals": 0,
        "away_goals": 0,
        "source_available_at_utc": "2026-09-21T12:00:00Z",
        "home_xg": 9.0,
        "away_xg": 0.01,
        "home_possession": 99.0,
        "away_possession": 1.0,
        "home_pass_accuracy": 99.0,
        "away_pass_accuracy": 1.0,
    })

    history = pd.DataFrame(history_rows)
    future = pd.DataFrame([{
        "match_id": "future",
        "competition": "EPL",
        "season": "2026/27",
        "season_start": 2026,
        "kickoff_utc": "2026-09-20T12:00:00Z",
        "home_team": "Alpha",
        "away_team": "Beta",
        "neutral_venue": False,
    }])

    out = build_match_features(history, future, windows=(3,))
    row = out.iloc[0]

    assert bool(row["pit_verified"]) is True
    assert row["home_xg_avg_3"] == 1.0
    assert row["away_xg_avg_3"] == 0.5
    assert row["xg_diff_3"] == 0.5
    assert row["possession_diff_3"] == 20.0
    assert row["pass_accuracy_diff_3"] == 5.0
    assert row["shots_inside_box_diff_3"] == 4.0
    assert row["blocked_shots_diff_3"] == -1.0
    # The 9/18 event was not available by the 9/20 cutoff, so its extreme xG/possession
    # values cannot contaminate the prediction-time 3-match window.
    assert row["home_xg_avg_3"] == 1.0
    assert row["possession_diff_3"] == 20.0
