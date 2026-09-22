import numpy as np
import pandas as pd

from src.evaluation.walk_forward import _lookup_context_weights, _routed_ensemble_proba, _routing_context


def test_routing_context_is_prediction_time_only_and_deterministic():
    frame = pd.DataFrame([
        {
            "competition": "EPL",
            "elo_diff": 220.0,
            "home_goal_total_avg_5": 3.0,
            "away_goal_total_avg_5": 2.8,
            "rest_diff_hours": 30.0,
            "neutral_venue_known": True,
            "neutral_venue": False,
        },
        {
            "competition": "AG_M",
            "elo_diff": -90.0,
            "rest_diff_hours": -10.0,
            "neutral_venue_known": False,
            "neutral_venue": pd.NA,
        },
    ])
    a = _routing_context(frame)
    b = _routing_context(frame)
    assert a["routing_context"].tolist() == b["routing_context"].tolist()
    assert "EPL|LARGE_HOME|HIGH|MISSING|HOME_MAJOR|HOME_AWAY" == a.loc[0, "routing_context"]
    assert "AG_M|AWAY|MISSING|MISSING|AWAY_SMALL|UNKNOWN" == a.loc[1, "routing_context"]


def test_routing_context_handles_missing_numeric_columns():
    frame = pd.DataFrame([{"competition": "EPL"}])
    out = _routing_context(frame)
    assert out.loc[0, "routing_strength_gap"] == "MISSING"
    assert out.loc[0, "routing_scoring_environment"] == "MISSING"
    assert out.loc[0, "routing_draw_environment"] == "MISSING"
    assert out.loc[0, "routing_rest"] == "MISSING"
    assert out.loc[0, "routing_venue"] == "UNKNOWN"
    assert isinstance(out.loc[0, "routing_context"], str)


class _FixedModel:
    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=float)

    def predict_proba(self, X):
        return np.repeat(self.probs[None, :], len(X), axis=0)


def test_batch_routed_ensemble_matches_rowwise_routing():
    frame = pd.DataFrame([
        {"competition": "EPL", "elo_diff": 220.0, "home_goal_total_avg_5": 3.0, "away_goal_total_avg_5": 2.8, "rest_diff_hours": 30.0, "neutral_venue_known": True, "neutral_venue": False, "f1": 1.0},
        {"competition": "EPL", "elo_diff": 20.0, "home_goal_total_avg_5": 2.0, "away_goal_total_avg_5": 2.0, "rest_diff_hours": 0.0, "neutral_venue_known": True, "neutral_venue": False, "f1": 2.0},
    ])
    models = {
        "a": _FixedModel([0.70, 0.20, 0.10]),
        "b": _FixedModel([0.20, 0.50, 0.30]),
    }
    context_weights = {
        "FULL:EPL|LARGE_HOME|HIGH|MISSING|HOME_MAJOR|HOME_AWAY": {"a": 0.8, "b": 0.2},
        "GLOBAL": {"a": 0.5, "b": 0.5},
    }
    fallback = {"a": 0.5, "b": 0.5}
    batch, routes = _routed_ensemble_proba(frame, models, ["f1"], context_weights, fallback)

    routed = _routing_context(frame)
    expected = []
    for _, row in routed.iterrows():
        local, route = _lookup_context_weights(row, context_weights, fallback)
        p = sum(local[name] * models[name].predict_proba(pd.DataFrame([row]))[0] for name in models)
        p = p / p.sum()
        expected.append(p)
    expected = np.asarray(expected)
    assert np.allclose(batch, expected, atol=1e-12)
    assert routes[0].startswith("FULL:")
    assert routes[1] == "GLOBAL"
