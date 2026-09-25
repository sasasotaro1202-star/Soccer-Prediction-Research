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
                "source_available_at_utc": kickoff + pd.Timedelta(hours=2),
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
    assert result["negative_binomial_status"].eq("PASS").all()
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
        "time_decay_score_logloss",
        "time_decay_over_2_5_logloss",
        "time_decay_over_2_5_brier",
        "time_decay_btts_logloss",
        "time_decay_btts_brier",
        "negative_binomial_score_logloss",
        "negative_binomial_over_2_5_logloss",
        "negative_binomial_over_2_5_brier",
        "negative_binomial_btts_logloss",
        "negative_binomial_btts_brier",
        "mean_score_risk",
        "high_risk_score_n",
        "high_risk_score_share",
        "high_risk_score_logloss",
        "high_risk_top3_score_hit_rate",
        "high_risk_over_2_5_logloss",
        "high_risk_over_2_5_brier",
        "high_risk_btts_logloss",
        "high_risk_btts_brier",
    ):
        assert result[col].notna().all()


def test_score_walk_forward_requires_result_publication_time():
    frame = _rows(24).drop(columns=["source_available_at_utc"])
    try:
        run_score_walk_forward(frame, min_train=10, oos_block=5)
    except ValueError as exc:
        assert "source_available_at_utc" in str(exc)
    else:
        raise AssertionError("score OOS must fail closed without publication-time evidence")


def test_score_risk_diagnostics_are_bounded():
    result = run_score_walk_forward(_rows(), min_train=10, oos_block=5)
    assert result["mean_score_risk"].between(0.0, 1.0).all()
    assert result["high_risk_score_share"].between(0.0, 1.0).all()
    assert (result["high_risk_score_n"] >= 0).all()
