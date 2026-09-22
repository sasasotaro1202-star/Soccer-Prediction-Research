import numpy as np
import pandas as pd
import pytest

from src.models.dixon_coles import (
    dixon_coles_distribution,
    dixon_coles_tau,
    fit_dixon_coles_model,
    predict_dixon_coles_distribution,
)


def _history(n=30):
    rows = []
    for i in range(n):
        rows.append(
            {
                "kickoff_utc": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=i),
                "home_team": "A" if i % 2 == 0 else "B",
                "away_team": "B" if i % 2 == 0 else "A",
                "home_goals": 1 if i % 4 else 0,
                "away_goals": 0 if i % 4 else 1,
                "pit_verified": True,
                "competition": "EPL",
            }
        )
    rows.append(
        {
            "kickoff_utc": pd.Timestamp("2024-12-01", tz="UTC"),
            "home_team": "A",
            "away_team": "B",
            "home_goals": 9,
            "away_goals": 9,
            "pit_verified": False,
            "competition": "EPL",
        }
    )
    return pd.DataFrame(rows)


def test_dixon_coles_tau_is_finite_for_low_scores():
    for h, a in ((0, 0), (0, 1), (1, 0), (1, 1), (2, 2)):
        assert np.isfinite(dixon_coles_tau(h, a, 1.4, 1.1, -0.08))


def test_dixon_coles_distribution_is_normalized():
    out = dixon_coles_distribution(1.4, 1.1, -0.08, max_goals=12)
    probs = np.asarray([p for _, _, p in out])
    assert len(out) == 169
    assert np.isfinite(probs).all()
    assert np.all(probs >= 0)
    assert probs.sum() == pytest.approx(1.0, abs=1e-10)


def test_dixon_coles_fit_uses_only_pit_verified_rows():
    model = fit_dixon_coles_model(_history())
    assert model["training_rows"] == 30
    assert np.isfinite(model["rho"])
    lo, hi = model["rho_bounds"]
    assert lo <= model["rho"] <= hi


def test_dixon_coles_prediction_is_probability_valid():
    model = fit_dixon_coles_model(_history())
    out = predict_dixon_coles_distribution(model, "A", "B", "EPL", max_goals=12)
    probs = np.asarray([p for _, _, p in out])
    assert probs.sum() == pytest.approx(1.0, abs=1e-10)
    assert np.isfinite(probs).all()


def test_dixon_coles_prediction_is_deterministic():
    model = fit_dixon_coles_model(_history())
    a = predict_dixon_coles_distribution(model, "A", "B", "EPL", max_goals=12)
    b = predict_dixon_coles_distribution(model, "A", "B", "EPL", max_goals=12)
    assert a == b
