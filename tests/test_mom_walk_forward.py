import numpy as np
import pandas as pd
import pytest

from src.evaluation.mom_walk_forward import (
    run_mom_walk_forward,
    split_mom_development_locked,
)
from src.models.mom_model import MOM_FEATURE_COLUMNS


def _rows(n_matches=18, players=6):
    rows = []
    base = pd.Timestamp("2024-01-01", tz="UTC")
    for m in range(n_matches):
        kickoff = base + pd.Timedelta(days=m)
        for p in range(players):
            rows.append({
                "match_id": f"m{m}",
                "player_id": f"p{m}_{p}",
                "kickoff_utc": kickoff,
                "feature_available_at_utc": kickoff - pd.Timedelta(hours=2),
                "pit_verified": True,
                "is_motm": int(p == 0),
                "recent_rating_ewm": 7.0 + 0.02 * (players - p) + 0.01 * m,
                "recent_minutes_ewm": 70 + p,
                "recent_goals_per90_ewm": 0.15 + 0.01 * (players - p),
                "recent_assists_per90_ewm": 0.08 + 0.005 * (players - p),
                "recent_xg_per90_ewm": 0.12 + 0.01 * (players - p),
                "recent_xa_per90_ewm": 0.10 + 0.005 * (players - p),
                "recent_key_passes_per90_ewm": 1.0 + 0.10 * (players - p),
                "recent_shots_per90_ewm": 1.5 + 0.15 * (players - p),
                "recent_starts_rate": 0.8 - 0.02 * p,
                "team_attack_strength": 1.0 + 0.01 * m,
                "opponent_defense_strength": 1.0,
                "position_attack_weight": 1.0 if p < 4 else 0.7,
                "days_rest": 6.0,
            })
    return pd.DataFrame(rows)


def test_mom_wfo_is_chronological_and_has_locked_holdout():
    metrics = run_mom_walk_forward(
        _rows(),
        n_blocks=6,
        locked_blocks=2,
        min_train_matches=5,
    )
    assert len(metrics) >= 3
    assert metrics["block"].tolist() == sorted(metrics["block"].tolist())
    assert metrics["train_matches"].is_monotonic_increasing
    assert (metrics["min_candidate_count"] >= 4).all()
    assert np.isfinite(metrics[["top1_hit_rate", "top4_hit_rate", "mrr", "ndcg_at_4", "logloss", "brier", "ece"]].to_numpy()).all()


def test_mom_wfo_rejects_non_pit_feature_timestamp():
    d = _rows()
    d.loc[0, "feature_available_at_utc"] = d.loc[0, "kickoff_utc"] + pd.Timedelta(minutes=1)
    with pytest.raises(RuntimeError, match="feature timestamp after kickoff"):
        run_mom_walk_forward(d, n_blocks=6, min_train_matches=5)


def test_mom_wfo_rejects_unknown_pit():
    d = _rows()
    d["pit_verified"] = pd.NA
    with pytest.raises(RuntimeError, match="unknown values"):
        run_mom_walk_forward(d, n_blocks=6, min_train_matches=5)


def test_mom_wfo_development_locked_split():
    block = pd.DataFrame({
        "block": [1, 2, 3, 4, 5, 6],
        "test_matches": [2] * 6,
        "top1_hit_rate": [0.1] * 6,
    })
    development, locked = split_mom_development_locked(block, locked_blocks=2)
    assert development["block"].tolist() == [1, 2, 3, 4]
    assert locked["block"].tolist() == [5, 6]


def test_mom_feature_schema_is_explicit():
    assert len(MOM_FEATURE_COLUMNS) >= 10
    assert all(isinstance(x, str) and x for x in MOM_FEATURE_COLUMNS)
