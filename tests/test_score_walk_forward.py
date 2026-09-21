import pandas as pd

from src.evaluation.score_walk_forward import run_score_walk_forward


def _rows(n=24):
    rows = []
    for i in range(n):
        kickoff = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=i)
        rows.append(
            {
                "match_id": f"m{i}",
                "kickoff_utc": kickoff,
                "home_team": "A" if i % 2 == 0 else "B",
                "away_team": "B" if i % 2 == 0 else "A",
                "home_goals": 1 if i % 3 else 2,
                "away_goals": 0 if i % 3 else 1,
                "pit_verified": True,
                "competition": "EPL",
            }
        )
    return pd.DataFrame(rows)


def test_score_walk_forward_produces_multiple_pit_safe_blocks():
    result = run_score_walk_forward(_rows(), min_train=10, oos_block=5)
    assert len(result) == 3
    assert int(result["n"].sum()) == 14
    for col in (
        "score_logloss",
        "exact_score_hit_rate",
        "top3_score_hit_rate",
        "top4_score_hit_rate",
        "home_goals_mae",
        "away_goals_mae",
        "total_goals_mae",
        "over_2_5_logloss",
        "over_2_5_brier",
        "btts_logloss",
        "btts_brier",
    ):
        assert result[col].notna().all()
