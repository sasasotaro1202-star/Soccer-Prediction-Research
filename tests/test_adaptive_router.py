import numpy as np
import pandas as pd
import pytest

from src.prediction.adaptive_router import AdaptiveModelRouter, RouterConfig

MODELS = ("logistic", "logistic_select", "extra_trees", "random_forest", "hist_gb")


def _oos(rows=240):
    out = []
    for i in range(rows):
        ts = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=i)
        y = i % 3
        probs = {
            "logistic": [0.55, 0.30, 0.15] if y == 0 else [0.20, 0.55, 0.25] if y == 1 else [0.15, 0.30, 0.55],
            "logistic_select": [0.50, 0.32, 0.18] if y == 0 else [0.24, 0.50, 0.26] if y == 1 else [0.18, 0.32, 0.50],
            "extra_trees": [0.48, 0.33, 0.19] if y == 0 else [0.23, 0.51, 0.26] if y == 1 else [0.19, 0.33, 0.48],
            "random_forest": [0.46, 0.34, 0.20] if y == 0 else [0.25, 0.49, 0.26] if y == 1 else [0.20, 0.34, 0.46],
            "hist_gb": [0.44, 0.35, 0.21] if y == 0 else [0.26, 0.48, 0.26] if y == 1 else [0.21, 0.35, 0.44],
        }
        for model, p in probs.items():
            out.append({
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
            })
    return pd.DataFrame(out)


def _ctx():
    return {
        "competition": "EPL",
        "strength_gap_bin": "MID",
        "scoring_environment_bin": "MID",
        "rest_bin": "NORMAL",
        "starter_status": "ANNOUNCED",
        "odds_missing": "FALSE",
    }


def test_pit_filtering():
    d = _oos()
    d.loc[d.index[-1], "outcome_available_at_utc"] = pd.Timestamp("2026-01-01", tz="UTC")
    router = AdaptiveModelRouter(config=RouterConfig(min_context_rows=10))
    router.fit(d, as_of="2025-08-30T00:00:00Z")
    assert all(np.isfinite(list(router._global_weights.values())))


def test_sparse_context_falls_back_to_global():
    router = AdaptiveModelRouter(config=RouterConfig(min_context_rows=500))
    router.fit(_oos())
    out = router.route(_ctx(), {m: np.array([[1 / 3, 1 / 3, 1 / 3]]) for m in MODELS})
    assert out["routing_source"] == "GLOBAL_OOS"
    assert np.isclose(sum(out["weights"].values()), 1.0)


def test_router_deterministic_and_valid():
    router = AdaptiveModelRouter(config=RouterConfig(min_context_rows=10))
    d = _oos()
    router.fit(d)
    probs = {m: np.array([[0.4, 0.3, 0.3], [0.2, 0.5, 0.3]]) for m in MODELS}
    a = router.route(_ctx(), probs)
    b = router.route(_ctx(), probs)
    assert np.allclose(a["probabilities"], b["probabilities"])
    assert np.allclose(a["probabilities"].sum(axis=1), 1.0)


def test_invalid_probabilities_fail_closed():
    router = AdaptiveModelRouter()
    router.fit(_oos())
    with pytest.raises(ValueError):
        router.route(
            _ctx(),
            {m: np.array([[0.7, 0.2, 0.1]]) for m in MODELS[:-1]} | {"hist_gb": np.array([[1.2, -0.1, -0.1]])},
        )


def test_missing_context_columns_do_not_crash():
    d = _oos().drop(columns=["starter_status", "odds_missing"])
    router = AdaptiveModelRouter(config=RouterConfig(min_context_rows=10))
    router.fit(d)
    out = router.route(
        {"competition": "EPL", "strength_gap_bin": "MID", "scoring_environment_bin": "MID", "rest_bin": "NORMAL"},
        {m: np.array([[0.5, 0.3, 0.2]]) for m in MODELS},
    )
    assert out["probabilities"].shape == (1, 3)
