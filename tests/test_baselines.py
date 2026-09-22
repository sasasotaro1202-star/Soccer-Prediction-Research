import numpy as np
import pandas as pd

from src.models.baselines import EloLogisticClassifier, RecencyLogisticClassifier, candidates


def _frame(n=30):
    return pd.DataFrame(
        {
            "elo_diff": np.linspace(-180, 180, n),
            "comp_elo_diff": np.linspace(-120, 120, n),
            "home_elo_expected": np.linspace(0.25, 0.75, n),
            "form": np.sin(np.arange(n) / 4.0),
        }
    )


def test_elo_candidate_produces_fixed_hda_probability_shape():
    X = _frame()
    y = np.array(([0, 1, 2] * 10), dtype=int)
    model = EloLogisticClassifier(random_state=42).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (30, 3)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-9)


def test_candidates_include_low_dimensional_elo_model():
    names = list(candidates(42))
    assert "elo_logistic" in names
    assert "recency_logistic" in names
    assert len(names) == 9
    assert "logistic_l2_strong" in names
    assert "hist_gb_robust" in names


def test_recency_logistic_is_probability_valid():
    X = _frame()
    y = np.array(([0, 1, 2] * 10), dtype=int)
    model = RecencyLogisticClassifier(random_state=42, half_life_rows=10).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (30, 3)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-9)
