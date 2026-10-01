from __future__ import annotations

import numpy as np
import pandas as pd

from src.prediction.model_bundle import load_bundle, predict_bundle, predict_bundle_with_diagnostics, train_and_save_bundle


def _training():
    rng = np.random.default_rng(7)
    n = 90
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    target = (f1 + 0.2 * f2 > 0).astype(int)
    target[:3] = [0, 1, 2]
    return pd.DataFrame({
        "f1": f1,
        "f2": f2,
        "kickoff_utc": pd.date_range("2026-01-01", periods=n, freq="h"),
        "pit_verified": True,
        "target": target,
    })


def test_shadow_diagnostics_match_prediction_and_are_bounded(tmp_path):
    df = _training()
    path = tmp_path / "bundle.pkl"
    train_and_save_bundle(
        df, ["f1", "f2"],
        {"weights": {"logistic": 1.0}, "temperature": 1.0},
        str(path), "diag-test", "snapshot",
    )
    bundle = load_bundle(str(path))
    X = df[["f1", "f2"]].iloc[:8]
    direct = predict_bundle(bundle, X)
    shadow, diagnostics = predict_bundle_with_diagnostics(bundle, X)

    assert np.allclose(shadow, direct, atol=1e-12)
    assert len(diagnostics["route"]) == len(X)
    for key in (
        "model_disagreement", "predictive_entropy", "uncertainty_score",
        "covariate_drift", "history_support_risk", "routing_risk",
    ):
        values = np.asarray(diagnostics[key], dtype=float)
        assert values.shape == (len(X),)
        assert np.all(np.isfinite(values))
        assert np.all((values >= 0.0) & (values <= 1.0))
    assert diagnostics["route"] == ["GLOBAL"] * len(X)


def test_shadow_diagnostics_reject_missing_features(tmp_path):
    df = _training()
    path = tmp_path / "bundle.pkl"
    train_and_save_bundle(
        df, ["f1", "f2"],
        {"weights": {"logistic": 1.0}, "temperature": 1.0},
        str(path), "diag-test", "snapshot",
    )
    bundle = load_bundle(str(path))
    import pytest
    try:
        predict_bundle_with_diagnostics(bundle, df[["f1"]].iloc[:1])
    except RuntimeError as exc:
        assert "missing" in str(exc).lower()
    else:
        raise AssertionError("missing feature diagnostics input was accepted")


def test_shadow_diagnostics_supports_static_contextual_routing(tmp_path):
    df = _training().copy()
    df["competition"] = np.where(np.arange(len(df)) % 2 == 0, "EPL", "LL")
    path = tmp_path / "bundle.pkl"
    train_and_save_bundle(
        df, ["f1", "f2"],
        {
            "weights": {"logistic": 1.0},
            "temperature": 1.0,
            "context_weights": {
                "COMP:EPL": {"logistic": 1.0},
                "COMP:LL": {"logistic": 1.0},
            },
            "contextual_temperatures": {"COMP:EPL": 1.0, "COMP:LL": 1.0, "GLOBAL": 1.0},
            "contextual_temperature_reasons": {
                "COMP:EPL": "test", "COMP:LL": "test", "GLOBAL": "test"
            },
        },
        str(path), "diag-test", "snapshot",
    )
    bundle = load_bundle(str(path))
    X = df.iloc[:8]
    direct = predict_bundle(bundle, X)
    shadow, diagnostics = predict_bundle_with_diagnostics(bundle, X)
    assert np.allclose(shadow, direct, atol=1e-12)
    assert len(diagnostics["route"]) == len(X)
    assert np.allclose(diagnostics["history_support_risk"], 0.0)
