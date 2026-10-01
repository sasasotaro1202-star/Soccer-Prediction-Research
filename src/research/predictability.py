"""Research-only Predictability Score and chronological calibration.

Predictability is distinct from forecast confidence: it estimates how structurally
forecastable a case looks from outcome-free diagnostics such as model disagreement,
uncertainty, covariate drift, and historical support.

Raw predictability is deterministic and target-free. Calibration estimates
P(forecast-correct | raw_predictability) using only chronologically prior OOS
outcomes. The locked OOS suffix is never fed back into the calibrator.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RAW_WEIGHTS = {
    "agreement": 0.30,
    "certainty": 0.30,
    "stability": 0.20,
    "support": 0.20,
}

def _vector(value: Any, n: int) -> np.ndarray:
    if value is None:
        return np.zeros(n, dtype=float)
    out = np.asarray(value, dtype=float).reshape(-1)
    if out.shape != (n,):
        raise ValueError("predictability diagnostic vector shape mismatch")
    if not np.isfinite(out).all():
        raise ValueError("predictability diagnostics contain non-finite values")
    if (out < 0.0).any() or (out > 1.0).any():
        raise ValueError("predictability diagnostics must be bounded in [0,1]")
    return out


def raw_predictability(
    *,
    disagreement: Any,
    uncertainty: Any,
    drift: Any,
    support_risk: Any,
) -> np.ndarray:
    arrays = [
        disagreement,
        uncertainty,
        drift,
        support_risk,
    ]
    lengths = [np.asarray(x).reshape(-1).shape[0] for x in arrays if x is not None]
    n = max(lengths, default=0)
    if n == 0:
        return np.empty(0, dtype=float)
    disagreement_v = _vector(disagreement, n)
    uncertainty_v = _vector(uncertainty, n)
    drift_v = _vector(drift, n)
    support_v = _vector(support_risk, n)
    score = (
        RAW_WEIGHTS["agreement"] * (1.0 - disagreement_v)
        + RAW_WEIGHTS["certainty"] * (1.0 - uncertainty_v)
        + RAW_WEIGHTS["stability"] * (1.0 - drift_v)
        + RAW_WEIGHTS["support"] * (1.0 - support_v)
    )
    return np.clip(score, 0.0, 1.0)


def fit_predictability_calibrator(
    history_raw: Any,
    history_correct: Any,
    current_raw: Any,
    *,
    min_rows: int = 120,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit calibration from prior outcomes only and apply to current raw scores."""
    raw_hist = np.asarray(history_raw, dtype=float).reshape(-1)
    y_hist = np.asarray(history_correct, dtype=int).reshape(-1)
    current = np.asarray(current_raw, dtype=float).reshape(-1)
    if len(raw_hist) != len(y_hist):
        raise ValueError("predictability history length mismatch")
    if not np.isfinite(current).all() or (current < 0.0).any() or (current > 1.0).any():
        raise ValueError("current predictability scores are invalid")
    if len(y_hist) < int(min_rows) or len(np.unique(y_hist)) < 2:
        prior = float(np.mean(y_hist)) if len(y_hist) else 0.5
        return np.full(len(current), np.clip(prior, 0.01, 0.99)), {
            "status": "FALLBACK_PRIOR_MEAN",
            "training_rows": int(len(y_hist)),
            "method": "prior_mean",
            "model": None,
        }
    if not np.isfinite(raw_hist).all() or (raw_hist < 0.0).any() or (raw_hist > 1.0).any():
        raise ValueError("predictability history scores are invalid")
    model = Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(C=0.50, max_iter=2000, random_state=13013)),
    ])
    model.fit(raw_hist.reshape(-1, 1), y_hist)
    calibrated = np.clip(
        model.predict_proba(current.reshape(-1, 1))[:, 1],
        0.01,
        0.99,
    )
    return calibrated, {
        "status": "FITTED_PRIOR_ONLY_LOGISTIC",
        "training_rows": int(len(y_hist)),
        "method": "prior_only_logistic",
        "model": model,
    }


def binary_calibration_metrics(correct: Any, probability: Any) -> dict[str, float]:
    y = np.asarray(correct, dtype=int).reshape(-1)
    p = np.asarray(probability, dtype=float).reshape(-1)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("predictability calibration arrays must have equal non-zero length")
    if not np.isfinite(p).all() or (p < 0.0).any() or (p > 1.0).any():
        raise ValueError("predictability probabilities are invalid")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("predictability calibration target must be binary")
    brier = float(np.mean((p - y) ** 2))
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for i in range(10):
        mask = (p >= bins[i]) & ((p < bins[i + 1]) if i < 9 else (p <= bins[i + 1]))
        if mask.any():
            ece += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return {
        "brier": brier,
        "ece": float(ece),
        "accuracy_at_0_5": float(((p >= 0.5).astype(int) == y).mean()),
    }
