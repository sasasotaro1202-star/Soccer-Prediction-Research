"""Research-only one-block-ahead future model-failure predictor.

The predictor estimates whether a candidate model will materially degrade on the
next chronological OOS block. Features are prediction-time aggregates from the
current block plus optional PIT-safe diagnostics. The failure label is created
only from the next block's matured outcomes. No production registry is touched.
"""
from __future__ import annotations

from typing import Mapping, Sequence
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-9


def _safe_probs(value) -> np.ndarray:
    p = np.asarray(value, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) == 0:
        raise ValueError("model probabilities must have shape (n,3)")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("model probabilities must be finite and non-negative")
    total = p.sum(axis=1, keepdims=True)
    if (total <= 0).any():
        raise ValueError("model probabilities contain zero-mass rows")
    return p / total


def _safe_auc(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(y) < 10 or len(np.unique(y)) < 2:
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def _brier(y, p):
    y = np.asarray(y, dtype=int)
    q = _safe_probs(p)
    onehot = np.zeros_like(q)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((q - onehot) ** 2, axis=1)))


def _logloss(y, p):
    y = np.asarray(y, dtype=int)
    q = np.clip(_safe_probs(p), EPS, 1.0)
    return float(-np.mean(np.log(q[np.arange(len(y)), y])))


def _entropy(p):
    q = np.clip(_safe_probs(p), EPS, 1.0)
    return -np.sum(q * np.log(q), axis=1)


def _block_features(
    fold: Mapping,
    name: str,
    candidate_names: Sequence[str],
) -> np.ndarray:
    p = _safe_probs(fold["preds"][name])
    mean_p = p
    confidence = mean_p.max(axis=1)
    ordered = np.sort(mean_p, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    entropy = _entropy(mean_p)

    others = []
    for other in candidate_names:
        if other == name or other not in (fold.get("preds") or {}):
            continue
        others.append(_safe_probs(fold["preds"][other]))
    if others:
        stack = np.stack(others, axis=0)
        disagreement = np.mean(np.std(np.concatenate([p[None, ...], stack], axis=0), axis=0), axis=1)
    else:
        disagreement = np.zeros(len(p), dtype=float)

    values = [
        confidence.mean(),
        margin.mean(),
        entropy.mean(),
        disagreement.mean(),
        float(len(p)),
    ]

    diagnostics = fold.get("diagnostics") or {}
    for key in (
        "model_disagreement",
        "predictive_entropy",
        "uncertainty_score",
        "covariate_drift",
        "history_support_risk",
        "routing_risk",
    ):
        value = diagnostics.get(key)
        if value is None:
            values.append(np.nan)
            continue
        arr = np.asarray(value, dtype=float)
        if arr.shape != (len(p),):
            raise ValueError(f"diagnostic {key!r} has invalid shape")
        if not np.isfinite(arr).all():
            raise ValueError(f"diagnostic {key!r} contains non-finite values")
        values.append(float(np.mean(np.clip(arr, 0.0, 1.0))))
    return np.asarray(values, dtype=float)


def evaluate_future_model_failure_oos(
    y: np.ndarray,
    names: Sequence[str],
    folds: Sequence[Mapping],
    *,
    min_training_transitions: int = 4,
    recent_window: int = 3,
    min_absolute_logloss_increase: float = 0.10,
) -> dict:
    """Cross-fit a one-block-ahead model-failure predictor.

    For transition k -> k+1, X comes only from block k predictions/diagnostics.
    The failure label for k+1 is based on block k+1 matured outcomes compared
    with the prior rolling loss baseline available before that block.
    """
    y = np.asarray(y, dtype=int)
    names = tuple(str(x) for x in names)
    if not names or len(folds) < 2:
        return {
            "status": "DEFERRED",
            "reason": "too_few_folds",
            "production_changed": False,
            "production_usable": False,
            "promotion_evidence_eligible": False,
        }

    histories = {name: [] for name in names}
    x_by_name = {name: [] for name in names}
    labels_by_name = {name: [] for name in names}
    predictions_by_name = {name: [] for name in names}
    transitions = []

    for k in range(len(folds) - 1):
        current = folds[k]
        future = folds[k + 1]
        start_k, end_k = int(current["end"]), int(current["te"])
        start_next, end_next = int(future["end"]), int(future["te"])
        yf = y[start_next:end_next]
        if len(yf) != end_next - start_next:
            raise ValueError("future fold target bounds are invalid")

        for name in names:
            if name not in (current.get("preds") or {}) or name not in (future.get("preds") or {}):
                raise ValueError(f"missing model predictions for {name!r}")
            current_p = _safe_probs(current["preds"][name])
            future_p = _safe_probs(future["preds"][name])
            if len(current_p) != end_k - start_k or len(future_p) != len(yf):
                raise ValueError(f"prediction length mismatch for {name!r}")

            current_p = _safe_probs(current["preds"][name])
            current_y = y[start_k:end_k]
            current_ll = _logloss(current_y, current_p)
            recent = histories[name][-int(recent_window):]
            baseline_history = list(recent) + [current_ll]
            baseline = float(np.median(baseline_history))
            next_ll = _logloss(yf, future_p)
            failure = bool(
                next_ll > baseline + float(min_absolute_logloss_increase)
            )
            if recent:
                failure = bool(
                    failure or next_ll > baseline * 1.10
                )

            x = _block_features(current, name, names)
            x_by_name[name].append(x)
            labels_by_name[name].append(int(failure))
            predictions_by_name[name].append(np.nan)
            histories[name].append(next_ll)

    for name in names:
        xs = np.asarray(x_by_name[name], dtype=float)
        ys = np.asarray(labels_by_name[name], dtype=int)
        p_out = np.full(len(ys), np.nan, dtype=float)
        for i in range(len(ys)):
            if i >= int(min_training_transitions) and len(np.unique(ys[:i])) >= 2:
                model = Pipeline([
                    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=2407,
                    )),
                ])
                model.fit(xs[:i], ys[:i])
                p_out[i] = float(np.clip(model.predict_proba(xs[i:i+1])[0, 1], 0.0, 1.0))
            elif i > 0:
                p_out[i] = float(np.mean(ys[:i]))
            else:
                p_out[i] = 0.0
        predictions_by_name[name] = p_out.tolist()

    metrics = {}
    for name in names:
        ys = np.asarray(labels_by_name[name], dtype=int)
        pp = np.asarray(predictions_by_name[name], dtype=float)
        usable = np.isfinite(pp)
        auc = _safe_auc(ys[usable], pp[usable])
        metrics[name] = {
            "transitions": int(len(ys)),
            "failure_rate": float(np.mean(ys)) if len(ys) else None,
            "failure_auc": auc,
            "predicted_failure_mean": float(np.mean(pp[usable])) if usable.any() else None,
            "production_usable": False,
        }

    return {
        "status": "EVALUATED",
        "models": metrics,
        "transitions": int(max(0, len(folds) - 2)),
        "horizon_blocks": 1,
        "label_rule": {
            "baseline_window": int(recent_window),
            "minimum_absolute_logloss_increase": float(min_absolute_logloss_increase),
            "secondary_relative_threshold": 0.10,
        },
        "production_changed": False,
        "production_usable": False,
        "promotion_evidence_eligible": False,
        "policy": "chronological_cross_fit_one_block_ahead_model_failure_prediction",
    }
