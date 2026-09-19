from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.evaluation.metrics import classification_metrics
from src.models.baselines import candidates

TARGET_ACCURACY = 0.80


def _fit_predict(model, fit, target):
    model.fit(fit, target)
    return model


def _blend_weights(scores: dict[str, dict]) -> dict[str, float]:
    names = list(scores)
    losses = np.array([max(float(scores[n]["logloss"]), 1e-6) for n in names], dtype=float)
    temperature = 0.08
    logits = -(losses - losses.min()) / temperature
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    floor = min(0.10, 1.0 / len(names))
    weights = np.maximum(weights, floor)
    weights /= weights.sum()
    return {name: float(w) for name, w in zip(names, weights)}


def _weighted_metric(parts: list[tuple[dict, int]], key: str) -> float:
    total = sum(n for _, n in parts)
    if not total:
        return float("nan")
    return float(sum(float(metric[key]) * n for metric, n in parts) / total)


def _temperature_transform(proba: np.ndarray, temperature: float) -> np.ndarray:
    p = np.clip(np.asarray(proba, dtype=float), 1e-9, 1.0)
    logits = np.log(p) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    return out / out.sum(axis=1, keepdims=True)


def _fit_temperature(y: pd.Series, proba: np.ndarray) -> tuple[float, bool]:
    """Fit one bounded temperature using strictly out-of-fit validation predictions."""
    yv = np.asarray(y, dtype=int)
    raw = np.clip(np.asarray(proba, dtype=float), 1e-9, 1.0)
    raw /= raw.sum(axis=1, keepdims=True)

    def objective(t: float) -> float:
        return float(classification_metrics(yv, _temperature_transform(raw, t))["logloss"])

    raw_loss = objective(1.0)
    result = minimize_scalar(objective, bounds=(0.70, 1.60), method="bounded", options={"xatol": 0.01})
    t = float(result.x) if result.success else 1.0
    improved = bool(result.success and objective(t) + 1e-6 < raw_loss)
    return (t if improved else 1.0), improved


def _validation_score(model, val: pd.DataFrame, feature_cols: list[str]) -> dict:
    if val.empty:
        raise ValueError("validation slice is empty")
    split = max(1, len(val) // 2)
    slices = [val.iloc[:split], val.iloc[split:]]
    parts = []
    for idx, sl in enumerate(slices):
        if sl.empty:
            continue
        metrics = classification_metrics(sl.target.astype(int), model.predict_proba(sl[feature_cols]))
        parts.append((metrics, len(sl) * (1 if idx == 0 else 2)))
    return {k: _weighted_metric(parts, k) for k in ("logloss", "accuracy", "brier", "rps", "ece")}


def run_walk_forward(df: pd.DataFrame, feature_cols: list[str], min_train: int = 1000, validation_frac: float = 0.2, oos_block: int | None = None, random_state: int = 42):
    """Chronological PIT-safe walk-forward with validation-only probability calibration."""
    if oos_block is None:
        oos_block = max(500, int(os.getenv("SOCCER_OOS_BLOCK", "2000")))
    validation_max = max(500, int(os.getenv("SOCCER_VALIDATION_MAX", "2000")))
    d = df.sort_values("kickoff_utc", kind="mergesort").reset_index(drop=True).copy()
    d = d[d["pit_verified"] == True].reset_index(drop=True).dropna(subset=["target"])
    if len(d) < min_train + oos_block:
        raise ValueError(f"Not enough PIT-verified rows: {len(d)}; need at least {min_train + oos_block}")

    results, selected = [], []
    start = min_train
    while start < len(d):
        oos_end = min(start + oos_block, len(d))
        train, oos = d.iloc[:start], d.iloc[start:oos_end]
        val_n = min(max(120, int(len(train) * validation_frac)), validation_max, max(120, len(train) - 300))
        fit, validation = train.iloc[:-val_n], train.iloc[-val_n:]
        # Keep model/ensemble selection and probability calibration on disjoint
        # chronological slices. Reusing the same rows for both creates a subtle
        # research-overfit channel even though neither touches the OOS block.
        split = len(validation) // 2
        if split < 60 or len(validation) - split < 60:
            raise ValueError("Validation slice must provide at least 60 rows for both selection and calibration")
        val_select = validation.iloc[:split]
        val_calib = validation.iloc[split:]

        # Fit each candidate only on the pre-validation fit slice for selection.
        validation_models = {}
        scores = {}
        for name, model in candidates(random_state).items():
            validation_models[name] = _fit_predict(model, fit[feature_cols], fit.target.astype(int))
            scores[name] = _validation_score(validation_models[name], val_select, feature_cols)
        weights = _blend_weights(scores)
        best = min(scores, key=lambda k: scores[k]["logloss"])

        # Calibration is learned from genuinely out-of-fit validation predictions.
        val_probs = np.zeros((len(val_calib), 3), dtype=float)
        for name, model in validation_models.items():
            val_probs += weights[name] * model.predict_proba(val_calib[feature_cols])
        val_probs = np.clip(val_probs, 1e-9, 1.0)
        val_probs /= val_probs.sum(axis=1, keepdims=True)
        calibration_temperature, calibration_used = _fit_temperature(val_calib.target.astype(int), val_probs)

        selected.append({"oos_start": str(oos.kickoff_utc.min()), "selected_model": best, "blend": "recent_weighted_two_slice_validation_softmax", "weights": weights, "validation_logloss": scores[best]["logloss"], "validation_accuracy": scores[best]["accuracy"], "validation_brier": scores[best]["brier"], "validation_rps": scores[best]["rps"], "validation_ece": scores[best]["ece"], "selection_rows": len(val_select), "calibration_rows": len(val_calib), "temperature": calibration_temperature, "temperature_calibration_used": calibration_used})

        # Refit candidates on all historical data available before this OOS block.
        fitted = {name: _fit_predict(model, train[feature_cols], train.target.astype(int)) for name, model in candidates(random_state).items()}
        probs = sum(weights[name] * model.predict_proba(oos[feature_cols]) for name, model in fitted.items())
        probs = np.clip(probs, 1e-9, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
        probs = _temperature_transform(probs, calibration_temperature)
        candidate_metrics = classification_metrics(oos.target.astype(int), probs)
        baseline_metrics = classification_metrics(oos.target.astype(int), fitted["logistic"].predict_proba(oos[feature_cols]))
        results.append({"oos_start": str(oos.kickoff_utc.min()), "oos_end": str(oos.kickoff_utc.max()), "leagues": "|".join(sorted(oos["competition"].astype(str).unique())), "seasons": "|".join(sorted(oos["season_start"].astype(str).unique())) if "season_start" in oos.columns else "", "model": "ensemble_calibrated", "best_single_model": best, **candidate_metrics, "baseline_logistic_logloss": baseline_metrics["logloss"], "baseline_logistic_accuracy": baseline_metrics["accuracy"], "baseline_logistic_brier": baseline_metrics["brier"], "baseline_logistic_rps": baseline_metrics["rps"], "baseline_logistic_ece": baseline_metrics["ece"], "n": len(oos), "target_accuracy": TARGET_ACCURACY, "target_met": bool(candidate_metrics["accuracy"] >= TARGET_ACCURACY), "target_gap": float(candidate_metrics["accuracy"] - TARGET_ACCURACY)})
        start = oos_end

    return pd.DataFrame(results), pd.DataFrame(selected)
