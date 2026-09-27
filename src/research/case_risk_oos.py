"""Chronological case-risk OOS layer for soccer.

The model estimates P(current baseline prediction is wrong). It does not alter
the incumbent probability by itself. Training for each OOS fold uses only
earlier OOS folds; frozen/holdout data are never consumed.
"""
from __future__ import annotations

from typing import Dict, Sequence
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EPS = 1e-6


def _clip(p):
    arr = np.asarray(p, dtype=float)
    return np.clip(arr, EPS, 1.0)


def _ensure_models(preds):
    arr = np.asarray(preds, dtype=float)
    if arr.ndim == 2:
        arr = arr[:, None, :]
    if arr.ndim != 3:
        raise ValueError("predictions must have shape (rows, classes) or (rows, models, classes)")
    return np.clip(arr, EPS, 1.0)


def _normalize(arr):
    arr = np.asarray(arr, dtype=float)
    denom = np.sum(arr, axis=-1, keepdims=True)
    return arr / np.maximum(denom, EPS)


def _entropy(p):
    q = _clip(_normalize(p))
    return -np.sum(q * np.log(q), axis=-1)


def _features(model_probs: np.ndarray, context: np.ndarray | None = None) -> np.ndarray:
    bp = _ensure_models(model_probs)
    bp = _normalize(bp)
    mean_p = np.mean(bp, axis=1)
    std_p = np.std(bp, axis=1) if bp.shape[1] > 1 else np.zeros_like(mean_p)
    order = np.sort(mean_p, axis=1)
    margin = order[:, -1] - order[:, -2] if mean_p.shape[1] >= 2 else np.ones(len(mean_p))
    top = np.max(mean_p, axis=1)
    base_entropy = _entropy(mean_p)
    disagreement = np.mean(std_p, axis=1)
    x = np.column_stack([mean_p, std_p, base_entropy, margin, top, disagreement])
    if context is not None:
        ctx = np.asarray(context, dtype=float)
        if ctx.ndim == 1:
            ctx = ctx.reshape(-1, 1)
        if len(ctx) != len(x):
            raise ValueError("context length mismatch")
        x = np.column_stack([x, ctx])
    return x


def _baseline_error_probability(model_probs: np.ndarray) -> np.ndarray:
    bp = _normalize(_ensure_models(model_probs))
    mean_p = np.mean(bp, axis=1)
    return np.clip(1.0 - np.max(mean_p, axis=1), EPS, 1.0 - EPS)


def _risk_coverage(errors, risk, coverages=(0.9, 0.8, 0.7, 0.6, 0.5)):
    errors = np.asarray(errors, dtype=int)
    risk = np.asarray(risk, dtype=float)
    order = np.argsort(risk, kind="mergesort")
    out = {}
    for coverage in coverages:
        k = max(1, int(round(len(errors) * float(coverage))))
        kept = errors[order[:k]]
        out[f"{float(coverage):.2f}"] = {
            "coverage": float(k / len(errors)),
            "error_rate": float(np.mean(kept)),
            "accuracy": float(1.0 - np.mean(kept)),
            "n": int(k),
        }
    return out


def _aurc(errors, risk):
    errors = np.asarray(errors, dtype=int)
    risk = np.asarray(risk, dtype=float)
    order = np.argsort(risk, kind="mergesort")
    kept = errors[order]
    cumulative = np.cumsum(kept) / np.arange(1, len(kept) + 1)
    coverage = np.arange(1, len(kept) + 1) / len(kept)
    if len(cumulative) <= 1:
        return float(cumulative[0]) if len(cumulative) else None
    return float(np.sum((cumulative[:-1] + cumulative[1:]) * np.diff(coverage) * 0.5))


def _safe_auc(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(y) < 20 or len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return None


def _safe_ap(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(y) < 20 or int(np.sum(y)) == 0:
        return None
    try:
        return float(average_precision_score(y, p))
    except ValueError:
        return None


def evaluate_case_risk_oof(
    y: np.ndarray,
    names: Sequence[str],
    folds: Sequence[dict],
    *,
    min_training_rows: int = 120,
) -> dict:
    y = np.asarray(y, dtype=int)
    meta_x = []
    meta_error = []
    pred_risk = []
    pred_base_risk = []
    fold_metrics = []

    for fold in folds:
        end = int(fold["end"])
        te = int(fold["te"])
        model_names = tuple(names)
        arrays = []
        for name in model_names:
            arrays.append(_ensure_models(np.asarray(fold["preds"][name], dtype=float)))
        if not arrays:
            raise ValueError("at least one model is required")
        bp = np.concatenate(arrays, axis=1)
        ctx = fold.get("context")
        x_fold = _features(bp, ctx)
        mean_p = np.mean(_normalize(bp), axis=1)
        base = np.argmax(mean_p, axis=1)
        yf = y[end:te]
        if len(yf) != len(base):
            raise ValueError("fold target length mismatch")
        errors = (base != yf).astype(int)

        risk = None
        if len(meta_x) >= min_training_rows and len(np.unique(meta_error)) >= 2:
            model = Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=2401)),
            ])
            try:
                model.fit(np.asarray(meta_x, dtype=float), np.asarray(meta_error, dtype=int))
                risk = np.clip(model.predict_proba(x_fold)[:, 1], EPS, 1.0 - EPS)
            except (ValueError, FloatingPointError):
                risk = None

        if risk is None:
            risk = _baseline_error_probability(bp)

        base_risk = _baseline_error_probability(bp)
        pred_risk.extend(risk.tolist())
        pred_base_risk.extend(base_risk.tolist())
        fold_metrics.append({
            "end": end,
            "te": te,
            "n": int(len(errors)),
            "risk_auc": _safe_auc(errors, risk),
            "risk_average_precision": _safe_ap(errors, risk),
            "baseline_confidence_auc": _safe_auc(errors, base_risk),
            "error_rate": float(np.mean(errors)),
        })
        meta_x.extend(x_fold.tolist())
        meta_error.extend(errors.tolist())

    if len(meta_error) < min_training_rows or len(np.unique(meta_error)) < 2:
        return {
            "status": "INSUFFICIENT_OOS",
            "oos_rows": int(len(meta_error)),
            "folds": int(len(fold_metrics)),
        }

    errors = np.asarray(meta_error, dtype=int)
    risk = np.asarray(pred_risk, dtype=float)
    base_risk = np.asarray(pred_base_risk, dtype=float)
    auc = _safe_auc(errors, risk)
    base_auc = _safe_auc(errors, base_risk)
    return {
        "status": "EVALUATED",
        "oos_rows": int(len(errors)),
        "folds": int(len(fold_metrics)),
        "risk_auc": auc,
        "risk_average_precision": _safe_ap(errors, risk),
        "baseline_confidence_auc": base_auc,
        "auc_improvement": None if auc is None or base_auc is None else float(auc - base_auc),
        "aurc": _aurc(errors, risk),
        "baseline_confidence_aurc": _aurc(errors, base_risk),
        "aurc_improvement": float(_aurc(errors, base_risk) - _aurc(errors, risk)),
        "risk_coverage": _risk_coverage(errors, risk),
        "baseline_confidence_coverage": _risk_coverage(errors, base_risk),
        "fold_metrics": fold_metrics,
        "policy": "research_only; chronological cross-fit error-risk; no holdout fitting",
    }


def fit_final_case_risk_model(y: np.ndarray, names: Sequence[str], folds: Sequence[dict]):
    y = np.asarray(y, dtype=int)
    x_all = []
    y_all = []
    for fold in folds:
        end, te = int(fold["end"]), int(fold["te"])
        arrays = [_ensure_models(np.asarray(fold["preds"][name], dtype=float)) for name in names]
        bp = np.concatenate(arrays, axis=1)
        x_all.extend(_features(bp, fold.get("context")).tolist())
        mean_p = np.mean(_normalize(bp), axis=1)
        y_all.extend((np.argmax(mean_p, axis=1) != y[end:te]).astype(int).tolist())
    if len(x_all) < 180 or len(np.unique(y_all)) < 2:
        return None
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=2401)),
    ])
    model.fit(np.asarray(x_all, dtype=float), np.asarray(y_all, dtype=int))
    return {"kind":"soccer_case_risk_logistic_v1","model":model,"training_rows":len(x_all),"policy":"pre_holdout_oos_only"}


def predict_case_risk(model_bundle, model_probs: np.ndarray, context: np.ndarray | None = None):
    if not model_bundle:
        return None
    x = _features(model_probs, context)
    return np.clip(model_bundle["model"].predict_proba(x)[:,1], EPS, 1.0-EPS)
