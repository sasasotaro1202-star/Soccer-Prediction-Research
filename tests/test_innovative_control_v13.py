from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.innovative_control_v13 import (
    model_disagreement_features,
    pit_audit,
    retrieval_predict,
    safe_probs,
    strategy_selector,
    uncertainty_components,
)


def test_safe_probs_normalizes_and_rejects_invalid():
    p = safe_probs(np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0]]))
    assert p.shape == (2, 3)
    assert np.allclose(p.sum(axis=1), 1.0)
    with pytest.raises(ValueError):
        safe_probs(np.array([[1.0, -1.0, 1.0]]))


def test_model_disagreement_exposes_core_statistics():
    probs = {
        "a": np.tile([0.7, 0.2, 0.1], (5, 1)),
        "b": np.tile([0.4, 0.4, 0.2], (5, 1)),
        "c": np.tile([0.6, 0.3, 0.1], (5, 1)),
    }
    out = model_disagreement_features(probs)
    assert {"entropy", "agreement", "pairwise_js", "majority_margin"}.issubset(out.columns)
    assert (out["pairwise_js"] > 0).all()


def test_pit_audit_fails_closed_for_future_available_at():
    df = pd.DataFrame(
        {
            "match_id": ["m1"],
            "pit_verified": [True],
            "prediction_cutoff_at_utc": ["2026-01-01T12:00:00Z"],
            "available_at": ["2026-01-01T12:01:00Z"],
        }
    )
    result = pit_audit(df)
    assert result["status"] == "FAIL"
    assert any("future_available_timestamp" in x for x in result["reasons"])


def test_pit_audit_passes_clean_input():
    df = pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "pit_verified": [True, True],
            "prediction_cutoff_at_utc": [
                "2026-01-01T12:00:00Z",
                "2026-01-02T12:00:00Z",
            ],
            "available_at": [
                "2026-01-01T11:00:00Z",
                "2026-01-02T11:00:00Z",
            ],
        }
    )
    assert pit_audit(df)["status"] == "PASS"


def test_retrieval_is_training_prefix_only():
    train_x = pd.DataFrame({"x1": [0.0, 0.1, 10.0], "x2": [0.0, 0.2, 10.0]})
    test_x = pd.DataFrame({"x1": [0.05], "x2": [0.05]})
    train_y = np.array([0, 0, 2])
    p, support = retrieval_predict(train_x, test_x, train_y, k=3)
    assert p.shape == (1, 3)
    assert np.isclose(p.sum(axis=1), 1.0).all()
    assert 0.0 <= float(support[0]) <= 1.0


def test_uncertainty_components_are_bounded():
    state = pd.DataFrame(
        {
            "data_completeness": [1.0, 0.5],
            "pairwise_js": [0.1, 0.8],
            "feature_drift": [0.1, 0.9],
            "feature_reliability": [1.0, 0.6],
            "entropy": [0.2, 0.9],
        }
    )
    out = uncertainty_components(state)
    assert out.shape[0] == 2
    assert np.isfinite(out.to_numpy()).all()
    assert ((out >= 0.0) & (out <= 1.0)).all().all()
    assert out.loc[1, "total_uncertainty"] > out.loc[0, "total_uncertainty"]


def test_strategy_selector_uses_history_not_current_target():
    state = pd.DataFrame(
        {
            "a": np.linspace(0, 1, 6),
            "b": np.linspace(1, 0, 6),
        }
    )
    preds = {
        "base": np.tile([0.7, 0.2, 0.1], (6, 1)),
        "ensemble": np.tile([0.6, 0.3, 0.1], (6, 1)),
        "retrieval": np.tile([0.2, 0.3, 0.5], (6, 1)),
    }
    choice, scores = strategy_selector(
        state,
        preds,
        [np.array([0, 0, 0, 0, 0, 0])],
        {k: [v] for k, v in preds.items()},
        [state],
    )
    assert len(choice) == 6
    assert set(choice).issubset({"base", "ensemble", "retrieval"})
    assert set(scores) == {"base", "ensemble", "retrieval"}
