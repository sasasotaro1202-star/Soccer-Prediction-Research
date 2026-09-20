import numpy as np
import pandas as pd
import pytest

from src.prediction.adaptive_router import AdaptiveModelRouter, RouterConfig


MODELS = ("ml_ensemble", "elo", "poisson", "market")


def _oos():
    rows = []
    for i in range(180):
        ts = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=i)
        y = i % 3
        probs = {
            "ml_ensemble": ([0.55, 0.30, 0.15] if y == 0 else [0.20, 0.55, 0.25] if y == 1 else [0.15, 0.30, 0.55]),
            "elo": ([0.50, 0.30, 0.20] if y == 0 else [0.25, 0.50, 0.25] if y == 1 else [0.20, 0.30, 0.50]),
            "poisson": ([0.48, 0.32, 0.20] if y == 0 else [0.24, 0.52, 0.24] if y == 1 else [0.20, 0.32, 0.48]),
            "market": [1 / 3, 1 / 3, 1 / 3],
        }
        for model, p in probs.items():
            rows.append(
                {
                    "prediction_time_utc": ts,
                    "outcome_available_at_utc": ts + pd.Timedelta(hours=2),
                    "y": y,
                    "model": model,
                    "p_home": p[0],
                    "p_draw": p[1],
                    "p_away": p[2],
                    "competition": "EPL",
                    "strength_gap_bin": "MID",
                    "scoring_environment_bin": "MID",
                    "rest_bin": "NORMAL",
                    "starter_status": "ANNOUNCED",
                    "odds_missing": "FALSE",
                }
            )
    return pd.DataFrame(rows)


def _ctx():
    return {
        "competition": "EPL",
        "strength_gap_bin": "MID",
        "scoring_environment_bin": "MID",
        "rest_bin": "NORMAL",
        "starter_status": "ANNOUNCED",
        "odds_missing": "FALSE",
    }


def test_router_requires_pit_safe_oos():
    d = _oos()
    d.loc[d.index[-1], "outcome_available_at_utc"] = pd.Timestamp("2026-01-01", tz="UTC")
    router = AdaptiveModelRouter(config=RouterConfig(min_regime_samples=10))
    router.fit(d, as_of="2025-06-29T00:00:00Z")
    out = router.route(
        _ctx(),
        {m: np.array([[0.5, 0.3, 0.2]]) for m in MODELS},
    )
    assert out["routing_source"] in {"GLOBAL_OOS", "REGIME_OOS"}


def test_sparse_regime_falls_back_to_global():
    router = AdaptiveModelRouter(config=RouterConfig(min_regime_samples=200))
    router.fit(_oos())
    out = router.route(_ctx(), {m: np.array([[0.7, 0.2, 0.1]]) for m in MODELS})
    assert out["routing_source"] == "GLOBAL_OOS"
    assert np.isclose(sum(out["weights"].values()), 1.0)


def test_routing_is_deterministic_and_probability_valid():
    router = AdaptiveModelRouter(config=RouterConfig(min_regime_samples=10))
    router.fit(_oos())
    probs = {m: np.array([[0.2, 0.5, 0.3], [0.4, 0.3, 0.3]]) for m in MODELS}
    a = router.route(_ctx(), probs)
    b = router.route(_ctx(), probs)
    assert np.allclose(a["probabilities"], b["probabilities"])
    assert a["probabilities"].shape == (2, 3)
    assert np.all((a["probabilities"] >= 0) & (a["probabilities"] <= 1))
    assert np.allclose(a["probabilities"].sum(axis=1), 1.0)


def test_invalid_candidate_probabilities_fail_closed():
    router = AdaptiveModelRouter()
    with pytest.raises(ValueError):
        router.route(
            {},
            {
                "ml_ensemble": [[0.5, 0.3, 0.2]],
                "elo": [[0.5, 0.3, 0.2]],
                "poisson": [[1.2, -0.1, -0.1]],
                "market": [[0.5, 0.3, 0.2]],
            },
        )


def test_binary_schema_is_rejected():
    router = AdaptiveModelRouter()
    d = _oos().drop(columns=["p_home", "p_draw", "p_away"])
    d["probability"] = 0.5
    with pytest.raises(ValueError):
        router.fit(d)
