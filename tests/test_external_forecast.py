import numpy as np
import pandas as pd
import pytest

from src.research.external_forecast import (
    ExternalForecastValidationError,
    build_diagnostics,
    disagreement_features,
    fit_linear_pool_weight,
    linear_pool,
    validate_external_forecasts,
)


def _frame(**overrides):
    base = {
        "match_id": ["m1", "m2"],
        "source": ["optA", "optA"],
        "prediction_time_utc": ["2026-09-01T12:00:00Z", "2026-09-02T12:00:00Z"],
        "source_available_at_utc": ["2026-09-01T09:00:00Z", "2026-09-02T09:00:00Z"],
        "p_home": [0.60, 0.20],
        "p_draw": [0.25, 0.30],
        "p_away": [0.15, 0.50],
        "provenance_url": ["https://example.test/a", "https://example.test/b"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_external_forecast_validation_is_pit_fail_closed():
    out = validate_external_forecasts(_frame())
    assert out["pit_status"].tolist() == ["PIT_VERIFIED", "PIT_VERIFIED"]


def test_unknown_availability_fails_closed():
    frame = _frame(source_available_at_utc=[None, "2026-09-02T09:00:00Z"])
    with pytest.raises(ExternalForecastValidationError, match="source_available_at_utc"):
        validate_external_forecasts(frame)


def test_future_available_source_fails_closed():
    frame = _frame(source_available_at_utc=["2026-09-01T13:00:00Z", "2026-09-02T09:00:00Z"])
    with pytest.raises(ExternalForecastValidationError, match="PIT"):
        validate_external_forecasts(frame)


def test_invalid_probabilities_fail_closed():
    frame = _frame(p_home=[0.9, 0.2])
    with pytest.raises(ExternalForecastValidationError, match="sum to 1"):
        validate_external_forecasts(frame)


def test_duplicate_identity_is_rejected():
    frame = pd.concat([_frame().iloc[[0]], _frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(ExternalForecastValidationError, match="duplicate"):
        validate_external_forecasts(frame)


def test_linear_pool_is_normalized_and_convex():
    base = np.array([[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]])
    ext = np.array([[0.2, 0.6, 0.2], [0.4, 0.4, 0.2]])
    fused = linear_pool(base, ext, 0.25)
    assert np.allclose(fused.sum(axis=1), 1.0)
    assert np.allclose(fused, 0.75 * base + 0.25 * ext)


def test_disagreement_features_capture_top_class_conflict():
    base = [[0.60, 0.25, 0.15]]
    ext = [[0.15, 0.70, 0.15]]
    out = disagreement_features(base, ext)
    assert bool(out.loc[0, "external_top_class_disagreement"]) is True
    assert np.isclose(out.loc[0, "external_disagreement_l1"], 0.90)


def test_weight_fit_uses_only_supplied_training_slice():
    base = np.tile([[0.70, 0.20, 0.10]], (20, 1))
    ext = np.tile([[0.15, 0.70, 0.15]], (20, 1))
    y = np.ones(20, dtype=int)
    result = fit_linear_pool_weight(y, base, ext)
    assert result["weight"] == 1.0
    assert result["external_logloss"] < result["base_logloss"]


def test_diagnostics_are_deterministic():
    d = build_diagnostics([[0.6, 0.2, 0.2]], [[0.2, 0.6, 0.2]])
    assert d.rows == 1
    assert d.source_count == 1
    assert d.pit_verified_rows == 1
    assert np.isclose(d.mean_disagreement_l1, 0.8)
    assert np.isclose(d.top_class_disagreement_rate, 1.0)
