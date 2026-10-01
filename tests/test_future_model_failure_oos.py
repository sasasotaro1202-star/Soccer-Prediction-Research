import numpy as np
import pandas as pd
import pytest

from src.research.future_model_failure_oos import evaluate_future_model_failure_oos


def _fold(start, size, base):
    n = size
    y = np.asarray([(i + start) % 3 for i in range(n)], dtype=int)
    preds = {}
    for name, shift in (("a", 0.0), ("b", 0.03), ("c", 0.06)):
        p = np.tile(np.array([[0.45, 0.35, 0.20]]), (n, 1))
        p[:, 0] = np.clip(p[:, 0] - shift + base, 0.05, 0.90)
        p[:, 1] = np.clip(p[:, 1] + shift, 0.05, 0.90)
        p[:, 2] = np.clip(1.0 - p[:, 0] - p[:, 1], 0.02, 0.90)
        p /= p.sum(axis=1, keepdims=True)
        preds[name] = p
    return {
        "end": start,
        "te": start + n,
        "preds": preds,
        "diagnostics": {
            "model_disagreement": np.full(n, 0.15),
            "predictive_entropy": np.full(n, 0.90),
            "uncertainty_score": np.full(n, 0.20),
            "covariate_drift": np.full(n, 0.10),
            "history_support_risk": np.full(n, 0.10),
            "routing_risk": np.full(n, 0.15),
        },
    }


def test_future_failure_predictor_is_chronological_and_research_only():
    folds = [_fold(i * 60, 60, 0.0) for i in range(7)]
    state = evaluate_future_model_failure_oos(
        np.asarray(sum([[(i % 3) for i in range(60)] for _ in range(7)], [])),
        ["a", "b", "c"],
        folds,
        min_training_transitions=3,
    )
    assert state["status"] == "EVALUATED"
    assert state["horizon_blocks"] == 1
    assert state["production_usable"] is False
    assert state["promotion_evidence_eligible"] is False
    assert state["transitions"] == 5


def test_future_failure_rejects_missing_model_prediction():
    folds = [_fold(i * 10, 10, 0.0) for i in range(2)]
    del folds[1]["preds"]["c"]
    y = np.zeros(20, dtype=int)
    with pytest.raises(ValueError, match="missing model predictions"):
        evaluate_future_model_failure_oos(y, ["a", "b", "c"], folds)


def test_too_few_folds_is_deferred():
    state = evaluate_future_model_failure_oos(np.zeros(5, dtype=int), ["a"], [_fold(0, 5, 0.0)])
    assert state["status"] == "DEFERRED"
    assert state["production_changed"] is False
