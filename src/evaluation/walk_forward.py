from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from src.evaluation.metrics import classification_metrics
from src.models.baselines import candidates

TARGET_ACCURACY = 0.80


def _fit_predict(model, fit, target):
    model.fit(fit, target)
    return model


def _blend_weights(scores: dict[str, dict]) -> dict[str, float]:
    """Deterministic baseline weights from per-model validation LogLoss."""
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


def _optimize_blend_weights(
    y: pd.Series | np.ndarray,
    model_probabilities: dict[str, np.ndarray],
    anchor: dict[str, float],
    *,
    min_weight: float = 0.05,
    max_weight: float = 0.70,
    regularization: float = 0.02,
) -> tuple[dict[str, float], bool]:
    """Optimize simplex weights against validation mixture LogLoss.

    The optimization is bounded and shrunk toward the deterministic anchor.
    It is run only on the model-selection slice, never on OOS or calibration rows.
    """
    names = list(model_probabilities)
    if not names:
        return {}, False
    arrays = []
    yv = np.asarray(y, dtype=int)
    for name in names:
        p = np.asarray(model_probabilities[name], dtype=float)
        if p.shape != (len(yv), 3):
            return dict(anchor), False
        if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
            return dict(anchor), False
        arrays.append(p)
    matrix = np.stack(arrays, axis=0)
    anchor_values = np.asarray([float(anchor.get(name, 0.0)) for name in names], dtype=float)
    if not np.isfinite(anchor_values).all() or (anchor_values < 0).any() or anchor_values.sum() <= 0:
        return dict(anchor), False
    anchor_values /= anchor_values.sum()
    if min_weight * len(names) > 1.0 or max_weight * len(names) < 1.0:
        return dict(anchor), False

    def objective(w: np.ndarray) -> float:
        blended = np.tensordot(w, matrix, axes=(0, 0))
        clipped = np.clip(blended, 1e-9, 1.0)
        clipped /= clipped.sum(axis=1, keepdims=True)
        loss = -np.log(clipped[np.arange(len(yv)), yv]).mean()
        penalty = regularization * float(np.sum((w - anchor_values) ** 2))
        return float(loss + penalty)

    bounds = [(float(min_weight), float(max_weight))] * len(names)
    constraints = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
    try:
        result = minimize(
            objective,
            x0=anchor_values,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-8, "disp": False},
        )
    except Exception:
        return dict(anchor), False
    if not result.success:
        return dict(anchor), False
    values = np.asarray(result.x, dtype=float)
    if not np.isfinite(values).all() or (values < min_weight - 1e-8).any() or (values > max_weight + 1e-8).any():
        return dict(anchor), False
    values = np.clip(values, min_weight, max_weight)
    values /= values.sum()
    return {name: float(w) for name, w in zip(names, values)}, True


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
    metrics = classification_metrics(val.target.astype(int), model.predict_proba(val[feature_cols]))
    return {k: metrics[k] for k in ("logloss", "accuracy", "brier", "rps", "ece")}


def _contextual_blend_weights(
    validation: pd.DataFrame,
    validation_models: dict,
    feature_cols: list[str],
    global_weights: dict[str, float],
    *,
    min_rows: int = 60,
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """Optimize mixture weights per competition with a conservative global fallback."""
    routed: dict[str, dict[str, float]] = {}
    reasons: dict[str, str] = {}
    if "competition" not in validation.columns:
        return {"__global__": global_weights}, {"__global__": "global_only"}

    for context, sl in validation.groupby("competition", sort=True):
        context_name = str(context)
        if len(sl) < min_rows:
            routed[context_name] = dict(global_weights)
            reasons[context_name] = "fallback_global_insufficient_validation_rows"
            continue
        probs = {name: model.predict_proba(sl[feature_cols]) for name, model in validation_models.items()}
        optimized, used = _optimize_blend_weights(
            sl.target.astype(int),
            probs,
            global_weights,
        )
        routed[context_name] = optimized if used else dict(global_weights)
        reasons[context_name] = "context_specific_optimized_validation" if used else "fallback_global_optimizer_failed"
    routed["__global__"] = dict(global_weights)
    reasons["__global__"] = "global_fallback"
    return routed, reasons


def run_walk_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_train: int = 1000,
    validation_frac: float = 0.2,
    oos_block: int | None = None,
    random_state: int = 42,
):
    """Chronological PIT-safe walk-forward with disjoint selection/calibration validation."""
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

        # Selection and calibration use disjoint chronological validation slices.
        split = len(validation) // 2
        if split < 60 or len(validation) - split < 60:
            raise ValueError("Validation slice must provide at least 60 rows for both selection and calibration")
        val_select = validation.iloc[:split]
        val_calib = validation.iloc[split:]

        validation_models = {}
        scores = {}
        for name, model in candidates(random_state).items():
            validation_models[name] = _fit_predict(model, fit[feature_cols], fit.target.astype(int))
            scores[name] = _validation_score(validation_models[name], val_select, feature_cols)

        anchor_weights = _blend_weights(scores)
        selection_probs = {
            name: model.predict_proba(val_select[feature_cols])
            for name, model in validation_models.items()
        }
        weights, blend_optimizer_used = _optimize_blend_weights(
            val_select.target.astype(int),
            selection_probs,
            anchor_weights,
        )
        best = min(scores, key=lambda k: scores[k]["logloss"])
        context_weights, context_reasons = _contextual_blend_weights(
            val_select,
            validation_models,
            feature_cols,
            weights,
        )

        # Probability calibration is fitted only on the second validation half.
        val_probs = np.zeros((len(val_calib), 3), dtype=float)
        if "competition" in val_calib.columns:
            for context, indices in val_calib.groupby("competition", sort=False).groups.items():
                positions = np.asarray(list(indices), dtype=int)
                local = context_weights.get(str(context), weights)
                sl = val_calib.loc[positions, feature_cols]
                for name, model in validation_models.items():
                    val_probs[positions] += local[name] * model.predict_proba(sl)
        else:
            for name, model in validation_models.items():
                val_probs += weights[name] * model.predict_proba(val_calib[feature_cols])
        val_probs = np.clip(val_probs, 1e-9, 1.0)
        val_probs /= val_probs.sum(axis=1, keepdims=True)
        calibration_temperature, calibration_used = _fit_temperature(val_calib.target.astype(int), val_probs)

        selected.append({
            "oos_start": str(oos.kickoff_utc.min()),
            "selected_model": best,
            "blend": "validation_optimized_contextual_ensemble",
            "weights": weights,
            "context_weights": context_weights,
            "context_reasons": context_reasons,
            "blend_optimizer_used": blend_optimizer_used,
            "validation_logloss": scores[best]["logloss"],
            "validation_accuracy": scores[best]["accuracy"],
            "validation_brier": scores[best]["brier"],
            "validation_rps": scores[best]["rps"],
            "validation_ece": scores[best]["ece"],
            "selection_rows": len(val_select),
            "calibration_rows": len(val_calib),
            "temperature": calibration_temperature,
            "temperature_calibration_used": calibration_used,
        })

        # Refit candidates on all historical data available before this OOS block.
        fitted = {
            name: _fit_predict(model, train[feature_cols], train.target.astype(int))
            for name, model in candidates(random_state).items()
        }
        probs = np.zeros((len(oos), 3), dtype=float)
        if "competition" in oos.columns:
            for context, indices in oos.groupby("competition", sort=False).groups.items():
                positions = np.asarray(list(indices), dtype=int)
                local = context_weights.get(str(context), weights)
                sl = oos.loc[positions, feature_cols]
                for name, model in fitted.items():
                    probs[positions] += local[name] * model.predict_proba(sl)
        else:
            for name, model in fitted.items():
                probs += weights[name] * model.predict_proba(oos[feature_cols])
        probs = np.clip(probs, 1e-9, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
        probs = _temperature_transform(probs, calibration_temperature)

        candidate_metrics = classification_metrics(oos.target.astype(int), probs)
        baseline_metrics = classification_metrics(
            oos.target.astype(int),
            fitted["logistic"].predict_proba(oos[feature_cols]),
        )
        results.append({
            "oos_start": str(oos.kickoff_utc.min()),
            "oos_end": str(oos.kickoff_utc.max()),
            "leagues": "|".join(sorted(oos["competition"].astype(str).unique())),
            "seasons": "|".join(sorted(oos["season_start"].astype(str).unique())) if "season_start" in oos.columns else "",
            "model": "ensemble_calibrated",
            "best_single_model": best,
            **candidate_metrics,
            "baseline_logistic_logloss": baseline_metrics["logloss"],
            "baseline_logistic_accuracy": baseline_metrics["accuracy"],
            "baseline_logistic_brier": baseline_metrics["brier"],
            "baseline_logistic_rps": baseline_metrics["rps"],
            "baseline_logistic_ece": baseline_metrics["ece"],
            "n": len(oos),
            "target_accuracy": TARGET_ACCURACY,
            "target_met": bool(candidate_metrics["accuracy"] >= TARGET_ACCURACY),
            "target_gap": float(candidate_metrics["accuracy"] - TARGET_ACCURACY),
            "blend_optimizer_used": blend_optimizer_used,
        })
        start = oos_end

    return pd.DataFrame(results), pd.DataFrame(selected)
