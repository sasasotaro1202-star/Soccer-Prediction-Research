import numpy as np
import pytest

from src.research.predictability import (
    fit_predictability_calibrator,
    raw_predictability,
)


def test_raw_predictability_is_distinct_from_confidence_and_bounded():
    out = raw_predictability(
        disagreement=np.array([0.0, 0.5, 1.0]),
        uncertainty=np.array([0.0, 0.5, 1.0]),
        drift=np.array([0.0, 0.5, 1.0]),
        support_risk=np.array([0.0, 0.5, 1.0]),
    )
    assert np.allclose(out, np.array([1.0, 0.5, 0.0]))
    assert np.isfinite(out).all()
    assert ((out >= 0.0) & (out <= 1.0)).all()


def test_calibrator_uses_only_prior_outcomes_and_falls_back_safely():
    current = np.array([0.2, 0.5, 0.8])
    calibrated, status = fit_predictability_calibrator(
        history_raw=np.array([0.2, 0.4]),
        history_correct=np.array([0, 1]),
        current_raw=current,
        min_rows=120,
    )
    assert np.isfinite(calibrated).all()
    assert (calibrated > 0).all() and (calibrated < 1).all()
    assert status["status"] == "FALLBACK_PRIOR_MEAN"
    assert status["training_rows"] == 2


def test_calibrator_fits_when_prior_history_is_sufficient():
    history_raw = np.concatenate([
        np.full(100, 0.15),
        np.full(100, 0.85),
    ])
    history_correct = np.concatenate([
        np.zeros(100, dtype=int),
        np.ones(100, dtype=int),
    ])
    current = np.array([0.15, 0.85])
    calibrated, status = fit_predictability_calibrator(
        history_raw=history_raw,
        history_correct=history_correct,
        current_raw=current,
    )
    assert status["status"] == "FITTED_PRIOR_ONLY_LOGISTIC"
    assert status["training_rows"] == 200
    assert calibrated[0] < calibrated[1]


def test_predictability_rejects_unbounded_inputs():
    with pytest.raises(ValueError, match="bounded"):
        raw_predictability(
            disagreement=[1.2],
            uncertainty=[0.0],
            drift=[0.0],
            support_risk=[0.0],
        )
