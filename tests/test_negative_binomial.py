import numpy as np
import pandas as pd

from src.models.negative_binomial import (
    fit_negative_binomial_score_model,
    negative_binomial_distribution,
    predict_negative_binomial_distribution,
)


def _history(n=80):
    rows = []
    for i in range(n):
        rows.append(
            {
                "kickoff_utc": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=i),
                "home_team": "A" if i % 2 == 0 else "B",
                "away_team": "B" if i % 2 == 0 else "A",
                "home_goals": [0, 1, 2, 1, 3, 2, 0, 4][i % 8],
                "away_goals": [1, 0, 1, 2, 0, 1, 3, 2][i % 8],
                "pit_verified": True,
                "competition": "EPL",
            }
        )
    return pd.DataFrame(rows)


def test_negative_binomial_distribution_is_finite_and_normalized():
    dist = negative_binomial_distribution(1.6, 1.2, 8.0, 10.0, max_goals=12)
    probs = np.asarray([p for _, _, p in dist], dtype=float)
    assert len(dist) == 169
    assert np.isfinite(probs).all()
    assert (probs >= 0).all()
    assert np.isclose(probs.sum(), 1.0, atol=1e-12)


def test_negative_binomial_model_is_pit_verified_only():
    history = _history()
    history.loc[0, "pit_verified"] = False
    model = fit_negative_binomial_score_model(history)
    assert model["training_rows"] == len(history) - 1
    assert model["method"] == "negative_binomial_pit_overdispersed_team_rates"
    assert np.isfinite(float(model["home_dispersion"]))
    assert np.isfinite(float(model["away_dispersion"]))


def test_negative_binomial_prediction_uses_pit_trained_model():
    model = fit_negative_binomial_score_model(_history())
    dist = predict_negative_binomial_distribution(model, "A", "B", "EPL", max_goals=10)
    probs = np.asarray([p for _, _, p in dist], dtype=float)
    assert np.isfinite(probs).all()
    assert np.isclose(probs.sum(), 1.0, atol=1e-12)
