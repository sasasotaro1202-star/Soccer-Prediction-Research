import numpy as np
import json

import pandas as pd
import pytest

from src.prediction.secondary_outputs import fit_score_rate_model, predict_mom_candidates, predict_score_candidates, predict_score_distribution, predict_score_markets


def test_score_model_is_pit_only_and_returns_exactly_three():
    history = pd.DataFrame(
        [
            {"home_team": "A", "away_team": "B", "home_goals": 2, "away_goals": 0, "pit_verified": True},
            {"home_team": "B", "away_team": "A", "home_goals": 0, "away_goals": 1, "pit_verified": True},
            {"home_team": "A", "away_team": "B", "home_goals": 9, "away_goals": 9, "pit_verified": False},
        ]
    )
    model = fit_score_rate_model(history)
    assert model["schema_version"] == 2
    assert model["method"] == "pit_smoothed_venue_split_team_goal_rates"
    assert model["teams"]["A"]["home_matches"] == 1.0
    assert len(predict_score_candidates(model, "A", "B")) == 3
    assert all(0 <= x["probability"] <= 1 for x in predict_score_candidates(model, "A", "B"))

def test_score_distribution_retains_low_tail_without_excessive_truncation():
    history = pd.DataFrame(
        [
            {"home_team": "A", "away_team": "B", "home_goals": 2, "away_goals": 0, "pit_verified": True},
            {"home_team": "B", "away_team": "A", "home_goals": 0, "away_goals": 1, "pit_verified": True},
            {"home_team": "A", "away_team": "C", "home_goals": 3, "away_goals": 1, "pit_verified": True},
            {"home_team": "C", "away_team": "B", "home_goals": 1, "away_goals": 1, "pit_verified": True},
        ]
    )
    model = fit_score_rate_model(history)
    distribution = predict_score_distribution(model, "A", "B", max_goals=12)
    assert len(distribution) == 169
    assert sum(p for _, _, p in distribution) == pytest.approx(1.0, abs=1e-10)
    assert all(np.isfinite(p) and p >= 0 for _, _, p in distribution)


def test_score_model_uses_shrunk_competition_environment():
    history = pd.DataFrame(
        [
            {"home_team": "A", "away_team": "B", "home_goals": 4, "away_goals": 0, "competition": "HIGH", "pit_verified": True},
            {"home_team": "B", "away_team": "A", "home_goals": 0, "away_goals": 1, "competition": "LOW", "pit_verified": True},
            {"home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0, "competition": "HIGH", "pit_verified": True},
            {"home_team": "B", "away_team": "A", "home_goals": 0, "away_goals": 1, "competition": "LOW", "pit_verified": True},
        ]
    )
    model = fit_score_rate_model(history)
    assert set(model["competition_rates"]) == {"HIGH", "LOW"}
    high = predict_score_candidates(model, "A", "B", "HIGH")
    low = predict_score_candidates(model, "A", "B", "LOW")
    assert high[0]["probability"] != low[0]["probability"]



def test_score_model_fails_closed_for_unknown_team():
    history = pd.DataFrame(
        [{"home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0, "pit_verified": True}]
    )
    model = fit_score_rate_model(history)
    with pytest.raises(RuntimeError):
        predict_score_candidates(model, "A", "UNKNOWN")


def test_mom_returns_exactly_four_without_renormalization():
    candidates = json.dumps([
        {"player_id": "p1", "probability": 0.10},
        {"player_id": "p2", "probability": 0.30},
        {"player_id": "p3", "probability": 0.05},
        {"player_id": "p4", "probability": 0.40},
        {"player_id": "p5", "probability": 0.15},
    ])
    result = predict_mom_candidates(candidates)
    assert len(result) == 4
    assert [x["player_id"] for x in result] == ["p4", "p2", "p5", "p1"]
    assert sum(x["probability"] for x in result) == pytest.approx(0.95)


def test_mom_requires_four_eligible_players():
    with pytest.raises(ValueError):
        predict_mom_candidates(json.dumps([
            {"player_id": "p1", "probability": 0.5},
            {"player_id": "p2", "probability": 0.5},
            {"player_id": "p3", "probability": 0.0},
        ]))


def test_score_markets_are_coherent_probabilities():
    history = pd.DataFrame(
        [
            {"home_team": "A", "away_team": "B", "home_goals": 2, "away_goals": 0, "pit_verified": True},
            {"home_team": "B", "away_team": "A", "home_goals": 0, "away_goals": 1, "pit_verified": True},
            {"home_team": "A", "away_team": "C", "home_goals": 3, "away_goals": 1, "pit_verified": True},
            {"home_team": "C", "away_team": "B", "home_goals": 1, "away_goals": 1, "pit_verified": True},
        ]
    )
    model = fit_score_rate_model(history)
    markets = predict_score_markets(model, "A", "B")
    for line in ("0_5", "1_5", "2_5", "3_5", "4_5"):
        assert markets[f"over_{line}"] + markets[f"under_{line}"] == pytest.approx(1.0, abs=1e-9)
        assert 0.0 <= markets[f"over_{line}"] <= 1.0
        assert 0.0 <= markets[f"under_{line}"] <= 1.0
    assert markets["btts_yes"] + markets["btts_no"] == pytest.approx(1.0, abs=1e-9)


def test_recency_score_model_is_order_sensitive_and_pit_only():
    rows = []
    base = pd.Timestamp("2025-01-01", tz="UTC")
    for i in range(12):
        rows.append({
            "kickoff_utc": base + pd.Timedelta(days=i),
            "home_team": "A",
            "away_team": "B",
            "home_goals": 0 if i < 6 else 3,
            "away_goals": 0 if i < 6 else 0,
            "pit_verified": True,
            "competition": "EPL",
        })
    rows.append({
        "kickoff_utc": base + pd.Timedelta(days=30),
        "home_team": "A",
        "away_team": "B",
        "home_goals": 9,
        "away_goals": 9,
        "pit_verified": False,
        "competition": "EPL",
    })
    from src.prediction.secondary_outputs import fit_recency_score_rate_model
    model = fit_recency_score_rate_model(pd.DataFrame(rows), half_life_rows=2.0)
    assert model["training_rows"] == 12
    recent_mean = model["home_mean"]
    assert recent_mean > 1.0
    assert recent_mean < 3.1


def test_time_decay_score_model_uses_elapsed_time_and_is_pit_only():
    rows = []
    base = pd.Timestamp("2025-01-01", tz="UTC")
    for days, goals in [(0, 9), (10, 3), (20, 1), (365, 0)]:
        rows.append({
            "kickoff_utc": base + pd.Timedelta(days=days),
            "home_team": "A",
            "away_team": "B",
            "home_goals": goals,
            "away_goals": 0,
            "pit_verified": True,
            "competition": "EPL",
        })
    rows.append({
        "kickoff_utc": base + pd.Timedelta(days=400),
        "home_team": "A",
        "away_team": "B",
        "home_goals": 50,
        "away_goals": 50,
        "pit_verified": False,
        "competition": "EPL",
    })
    from src.prediction.secondary_outputs import fit_time_decay_score_rate_model
    model = fit_time_decay_score_rate_model(pd.DataFrame(rows), half_life_days=20.0)
    assert model["training_rows"] == 4
    assert model["half_life_days"] == 20.0
    assert model["home_mean"] < 3.0


def test_time_decay_dispatch_metadata_is_distinct():
    rows = [{
        "kickoff_utc": pd.Timestamp("2025-01-01", tz="UTC"),
        "home_team": "A",
        "away_team": "B",
        "home_goals": 1,
        "away_goals": 0,
        "pit_verified": True,
        "competition": "EPL",
    }]
    from src.prediction.secondary_outputs import fit_time_decay_score_rate_model
    model = fit_time_decay_score_rate_model(pd.DataFrame(rows))
    assert model["method"] == "pit_time_decay_weighted_venue_split_team_goal_rates"


def test_score_distribution_dispatches_dixon_coles_method():
    history = pd.DataFrame(
        [
            {"home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0, "kickoff_utc": "2025-01-01", "pit_verified": True},
            {"home_team": "B", "away_team": "A", "home_goals": 0, "away_goals": 1, "kickoff_utc": "2025-01-02", "pit_verified": True},
            {"home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 1, "kickoff_utc": "2025-01-03", "pit_verified": True},
        ]
    )
    from src.models.dixon_coles import fit_dixon_coles_model
    model = fit_dixon_coles_model(history)
    dist = predict_score_distribution(model, "A", "B", "EPL", max_goals=5)
    assert len(dist) == 36
    assert abs(sum(p for _, _, p in dist) - 1.0) < 1e-9
    assert all(np.isfinite(p) and p >= 0 for _, _, p in dist)
