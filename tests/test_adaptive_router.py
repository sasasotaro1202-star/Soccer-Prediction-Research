import numpy as np
import pandas as pd
import pytest

from src.prediction.adaptive_router import AdaptiveModelRouter, RouterConfig


def _oos():
    rows = []
    for i in range(180):
        ts = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=i)
        y = i % 3
        for model, p in (
            ("ml_ensemble", 0.55 if y else 0.45),
            ("elo", 0.52 if y else 0.48),
            ("poisson", 0.51 if y else 0.49),
            ("market", 0.50 if y else 0.50),
        ):
            rows.append(
                {
                    "prediction_time_utc": ts,
                    "outcome_available_at_utc": ts + pd.Timedelta(hours=2),
                    "y": y,
                    "model": model,
                    "p_home": p[0],\n                    "p_draw": p[1],\n                    "p_away": p[2],
                    "competition": "EPL",
                    "strength_gap_bin": "MID",
                    "scoring_environment_bin": "MID",
                    "rest_bin": "NORMAL",
                    "starter_status": "ANNOUNCED",
                    "odds_missing": "FALSE",
                }
            )
    return pd.DataFrame(rows)


def test_router_requires_pit_safe_oos():
    d = _oos()
    d.loc[d.index[-1], "outcome_available_at_utc"] = pd.Timestamp("2026-01-01", tz="UTC")
    router = AdaptiveModelRouter(config=RouterConfig(min_regime_samples=10))
    router.fit(d, as_of="2025-06-29T00:00:00Z")
    out = router.route(
        {c: "MID" for c in ("competition", "strength_gap_bin", "scoring_environment_bin", "rest_bin")},
        {m: np.array([[0.5, 0.3, 0.2]]) for m in ("ml_ensemble", "elo", "poisson", "market")},
    )
    assert out["routing_source"] in {"GLOBAL_OOS", "REGIME_OOS"}


def test_sparse_regime_falls_back_to_global():
    router = AdaptiveModelRouter(config=RouterConfig(min_regime_samples=200))
    router.fit(_oos())
    out = router.route(
        {
            "competition": "EPL",
            "strength_gap_bin": "MID",
            "scoring_environment_bin": "MID",
            "rest_bin": "NORMAL",
            "starter_status": "ANNOUNCED",
            "odds_missing": "FALSE",
        },
        {m: np.array([[0.7, 0.2, 0.1]]) for m in ("ml_ensemble", "elo", "poisson", "market")},
    )
    assert out["routing_source"] == "GLOBAL_OOS"
    assert np.isclose(sum(out["weights"].values()), 1.0)


def test_routing_is_deterministic_and_probability_valid():
    router = AdaptiveModelRouter(config=RouterConfig(min_regime_samples=10))
    router.fit(_oos())
    ctx = {
        "competition": "EPL",
        "strength_gap_bin": "MID",
        "scoring_environment_bin": "MID",
        "rest_bin": "NORMAL",
        "starter_status": "ANNOUNCED",
        "odds_missing": "FALSE",
    }
    probs = {m: np.array([[0.2, 0.5, 0.3], [0.4, 0.3, 0.3]]) for m in ("ml_ensemble", "elo", "poisson", "market")}
    a = router.route(ctx, probs)
    b = router.route(ctx, probs)
    assert np.allclose(a["probabilities"], b["probabilities"])
    assert np.all((a["probabilities"] >= 0) & (a["probabilities"] <= 1))


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
