import numpy as np
import pandas as pd

from src.monitoring.drift import feature_drift, prediction_drift, psi


def test_psi_is_zero_for_identical_distribution():
    x = np.linspace(0, 1, 100)
    assert abs(psi(x, x)) < 1e-12


def test_feature_drift_uses_shared_numeric_columns():
    ref = pd.DataFrame({"x": np.arange(100), "name": ["a"] * 100})
    cur = pd.DataFrame({"x": np.arange(100) + 1, "name": ["b"] * 100})
    out = feature_drift(ref, cur)
    assert list(out["feature"]) == ["x"]
    assert out.loc[0, "reference_n"] == 100
    assert out.loc[0, "current_n"] == 100


def test_prediction_drift_reports_insufficient_sample():
    ref = pd.DataFrame({"p_home": [0.5] * 10})
    cur = pd.DataFrame({"p_home": [0.6] * 10})
    out = prediction_drift(ref, cur)
    assert out["status"] == "OK"
    assert out["columns"]["p_home"]["status"] == "INSUFFICIENT_SAMPLE"
