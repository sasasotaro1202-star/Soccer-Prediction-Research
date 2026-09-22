import numpy as np
import pandas as pd

from src.evaluation.walk_forward import _contextual_temperatures, _apply_contextual_temperatures, _lookup_context_weights, _routed_ensemble_proba, _routing_context


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


def test_routing_fallback_preserves_draw_environment_before_competition():
    frame = pd.DataFrame([{
        "competition": "EPL",
        "elo_diff": 20.0,
        "home_goal_total_avg_5": 2.0,
        "away_goal_total_avg_5": 2.0,
        "home_draw_rate_20": 0.38,
        "away_draw_rate_20": 0.36,
        "rest_diff_hours": 0.0,
        "neutral_venue_known": True,
        "neutral_venue": False,
        "f1": 1.0,
    }])
    models = {
        "a": _FixedModel([0.80, 0.10, 0.10]),
        "b": _FixedModel([0.10, 0.80, 0.10]),
    }
    context_weights = {
        "COMP_DRAW:EPL|HIGH": {"a": 0.1, "b": 0.9},
        "COMP:EPL": {"a": 0.9, "b": 0.1},
        "GLOBAL": {"a": 0.5, "b": 0.5},
    }
    batch, routes = _routed_ensemble_proba(frame, models, ["f1"], context_weights, {"a": 0.5, "b": 0.5})
    assert routes[0] == "COMP_DRAW:EPL|HIGH"
    assert np.argmax(batch[0]) == 1


def test_contextual_temperature_uses_global_fallback_for_sparse_routes():
    proba = np.tile(np.array([0.70, 0.20, 0.10]), (10, 1))
    y = pd.Series([0, 1, 0, 2, 0, 1, 0, 2, 1, 0])
    temps, reasons = _contextual_temperatures(
        y,
        proba,
        ["GLOBAL"] * 10,
        1.0,
        min_rows=6,
    )
    assert temps["GLOBAL"] == 1.0
    assert reasons["GLOBAL"] == "global_fallback"
    assert np.allclose(_apply_contextual_temperatures(proba, ["GLOBAL"] * 10, temps, 1.0), proba)


def test_contextual_temperature_is_shrunk_and_applies_by_route():
    rng = np.random.default_rng(42)
    raw = np.tile(np.array([0.85, 0.10, 0.05]), (80, 1))
    raw = raw + rng.normal(0.0, 0.002, raw.shape)
    raw = np.clip(raw, 1e-6, 1.0)
    raw /= raw.sum(axis=1, keepdims=True)
    y = pd.Series([0, 1, 2, 0] * 20)
    routes = ["COMP:EPL"] * 80
    temps, reasons = _contextual_temperatures(y, raw, routes, 1.0, min_rows=30, prior_strength=60)
    assert "COMP:EPL" in temps
    assert 0.70 <= temps["COMP:EPL"] <= 1.60
    assert reasons["COMP:EPL"] == "context_temperature_shrunk_from_calibration"
    calibrated = _apply_contextual_temperatures(raw, routes, temps, 1.0)
    assert calibrated.shape == raw.shape
    assert np.isfinite(calibrated).all()
    assert np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-12)
