import numpy as np
import pandas as pd

from src.models.baselines import HierarchicalEloLogisticClassifier, DynamicEloLogisticClassifier, EloLogisticClassifier, RecencyLogisticClassifier, QuantileLogisticClassifier, candidates


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
    assert len(names) == 11
    assert "logistic_l2_strong" in names
    assert "hist_gb_robust" in names
    assert "quantile_logistic" in names
    assert "dynamic_elo_logistic" in names
    assert "hierarchical_elo_logistic" in names


def test_recency_logistic_is_probability_valid():
    X = _frame()
    y = np.array(([0, 1, 2] * 10), dtype=int)
    model = RecencyLogisticClassifier(random_state=42, half_life_rows=10).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (30, 3)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-9)


def test_quantile_logistic_is_probability_valid():
    X = _frame()
    y = np.array(([0, 1, 2] * 10), dtype=int)
    model = QuantileLogisticClassifier(random_state=42, n_quantiles=16).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (30, 3)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-9)


def test_dynamic_elo_logistic_is_probability_valid():
    X = _frame()
    X["dynamic_elo_diff"] = X["elo_diff"] * 1.15
    X["dynamic_home_elo_expected"] = np.linspace(0.22, 0.78, len(X))
    y = np.array(([0, 1, 2] * 10), dtype=int)
    model = DynamicEloLogisticClassifier(random_state=42).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (30, 3)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-9)


def test_hierarchical_elo_logistic_is_probability_valid():
    X = _frame()
    X["comp_elo_shrunk_diff"] = X["elo_diff"] * 0.8
    X["home_comp_elo_shrunk_expected"] = np.linspace(0.30, 0.70, len(X))
    X["dynamic_elo_diff"] = X["elo_diff"] * 1.1
    y = np.array(([0, 1, 2] * 10), dtype=int)
    model = HierarchicalEloLogisticClassifier(random_state=42).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (30, 3)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-9)
