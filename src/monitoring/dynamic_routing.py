from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-9
LOG2 = float(np.log(2.0))
LOG3 = float(np.log(3.0))


def build_drift_reference(frame: pd.DataFrame, feature_cols: list[str], *, min_scale: float = 1e-6) -> dict[str, Any]:
    """Build a PIT-safe robust reference from rows available before inference."""
    reference: dict[str, Any] = {}
    for feature in feature_cols:
        if feature not in frame.columns:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        median = float(values.median())
        mad = float(np.median(np.abs(values.to_numpy(dtype=float) - median)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale < min_scale:
            std = float(values.std(ddof=0))
            scale = std if np.isfinite(std) and std >= min_scale else 1.0
        reference[str(feature)] = {"median": median, "scale": float(scale), "n": int(len(values))}
    return {"schema_version": 1, "type": "reference_only_robust_numeric", "features": reference}


def compute_drift_scores(frame: pd.DataFrame, reference: dict[str, Any], feature_cols: list[str]) -> np.ndarray:
    """Compute bounded row-wise covariate-shift scores from reference-only statistics."""
    ref_features = reference.get("features", {}) if isinstance(reference, dict) else {}
    n = len(frame)
    if n == 0:
        return np.empty(0, dtype=float)
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        z_values: list[float] = []
        missing = 0
        considered = 0
        for feature in feature_cols:
            stats = ref_features.get(str(feature))
            considered += 1
            if not isinstance(stats, dict) or feature not in frame.columns:
                missing += 1
                continue
            value = pd.to_numeric(pd.Series([frame.iloc[i][feature]]), errors="coerce").iloc[0]
            if not np.isfinite(value):
                missing += 1
                continue
            median = float(stats.get("median", 0.0))
            scale = max(abs(float(stats.get("scale", 1.0))), 1e-6)
            z_values.append(abs((float(value) - median) / scale))
        mean_z = float(np.mean(z_values)) if z_values else 0.0
        missing_rate = float(missing / considered) if considered else 0.0
        scores[i] = float(np.clip(0.75 * (mean_z / 3.0) + 0.25 * missing_rate, 0.0, 1.0))
    return scores


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("Soccer probability matrix must have shape (n, 3)")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Probability matrix contains invalid values")
    row_sum = p.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise ValueError("Probability matrix contains zero-sum rows")
    p = np.clip(p / row_sum, EPS, 1.0)
    return np.clip(-np.sum(p * np.log(p), axis=1) / LOG3, 0.0, 1.0)


def mean_js_disagreement(model_probabilities: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    names = list(model_probabilities)
    if not names:
        raise ValueError("No model probabilities supplied")
    arrays = []
    for name in names:
        p = np.asarray(model_probabilities[name], dtype=float)
        if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or (p < 0).any():
            raise ValueError(f"Invalid probability matrix for model {name!r}")
        sums = p.sum(axis=1, keepdims=True)
        if np.any(sums <= 0):
            raise ValueError(f"Model {name!r} produced zero-sum probability rows")
        arrays.append(p / sums)
    n = len(arrays[0])
    if any(len(p) != n for p in arrays):
        raise ValueError("Model probability row counts differ")
    mixture = np.mean(np.stack(arrays, axis=0), axis=0)
    mixture = np.clip(mixture, EPS, 1.0)
    mixture /= mixture.sum(axis=1, keepdims=True)
    js_values = []
    for p in arrays:
        p = np.clip(p, EPS, 1.0)
        p /= p.sum(axis=1, keepdims=True)
        m = np.clip((p + mixture) / 2.0, EPS, 1.0)
        js = 0.5 * np.sum(p * np.log(p / m), axis=1) + 0.5 * np.sum(mixture * np.log(mixture / m), axis=1)
        js_values.append(js)
    disagreement = np.mean(np.stack(js_values, axis=0), axis=0) / LOG2
    return np.clip(disagreement, 0.0, 1.0), mixture


def dynamic_route_weights(
    base_weight_matrix: np.ndarray,
    fallback_weights: np.ndarray | list[float],
    model_probabilities: dict[str, np.ndarray],
    drift_scores: np.ndarray,
    *,
    drift_strength: float = 0.85,
    uncertainty_strength: float = 0.75,
    min_specialist_trust: float = 0.25,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Move contextual weights toward the global fallback when risk rises."""
    base = np.asarray(base_weight_matrix, dtype=float)
    if base.ndim != 2 or not np.isfinite(base).all() or (base < 0).any():
        raise ValueError("base_weight_matrix is invalid")
    names = list(model_probabilities)
    if base.shape[1] != len(names) or len(base) != len(drift_scores):
        raise ValueError("Dynamic routing dimensions do not match")
    fallback = np.asarray(fallback_weights, dtype=float)
    if fallback.ndim != 1 or fallback.shape[0] != len(names) or not np.isfinite(fallback).all() or (fallback < 0).any() or fallback.sum() <= 0:
        raise ValueError("fallback_weights are invalid")
    fallback /= fallback.sum()
    row_sums = base.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("base routing contains zero-sum rows")
    base /= row_sums
    drift = np.asarray(drift_scores, dtype=float)
    if not np.isfinite(drift).all() or (drift < 0).any():
        raise ValueError("drift_scores are invalid")
    drift = np.clip(drift, 0.0, 1.0)
    disagreement, mixture = mean_js_disagreement(model_probabilities)
    entropy = normalized_entropy(mixture)
    uncertainty = np.clip(0.55 * entropy + 0.45 * disagreement, 0.0, 1.0)
    if not np.isfinite(drift_strength) or not 0.0 <= float(drift_strength) <= 3.0:
        raise ValueError("drift_strength must be within [0, 3]")
    if not np.isfinite(uncertainty_strength) or not 0.0 <= float(uncertainty_strength) <= 3.0:
        raise ValueError("uncertainty_strength must be within [0, 3]")
    if not np.isfinite(min_specialist_trust) or not 0.05 <= float(min_specialist_trust) <= 0.95:
        raise ValueError("min_specialist_trust must be within [0.05, 0.95]")
    trust = np.exp(-float(drift_strength) * drift - float(uncertainty_strength) * uncertainty)
    trust = np.clip(trust, float(min_specialist_trust), 1.0)
    dynamic = fallback[None, :] + trust[:, None] * (base - fallback[None, :])
    dynamic = np.clip(dynamic, 0.0, None)
    sums = dynamic.sum(axis=1, keepdims=True)
    if np.any(sums <= 0) or not np.isfinite(sums).all():
        raise ValueError("Dynamic routing produced invalid weights")
    dynamic /= sums
    return dynamic, {"trust": trust, "uncertainty": uncertainty, "entropy": entropy, "disagreement": disagreement, "drift": drift}
