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

def _routing_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive low-cardinality prediction-time context bins without using outcomes."""
    d = frame.copy()

    if "elo_diff" in d.columns:
        elo = pd.to_numeric(d["elo_diff"], errors="coerce")
        d["routing_strength_gap"] = pd.cut(
            elo,
            bins=[-np.inf, -200.0, -75.0, 75.0, 200.0, np.inf],
            labels=["LARGE_AWAY", "AWAY", "EVEN", "HOME", "LARGE_HOME"],
        ).astype("string")
    else:
        d["routing_strength_gap"] = "MISSING"

    home_goal = pd.to_numeric(
        d["home_goal_total_avg_5"] if "home_goal_total_avg_5" in d.columns else pd.Series(np.nan, index=d.index),
        errors="coerce",
    )
    away_goal = pd.to_numeric(
        d["away_goal_total_avg_5"] if "away_goal_total_avg_5" in d.columns else pd.Series(np.nan, index=d.index),
        errors="coerce",
    )
    goal_env = (home_goal + away_goal) / 2.0
    d["routing_scoring_environment"] = pd.cut(
        goal_env,
        bins=[-np.inf, 1.8, 2.3, 2.8, np.inf],
        labels=["LOW", "MID_LOW", "MID_HIGH", "HIGH"],
    ).astype("string").fillna("MISSING")

    home_draw = pd.to_numeric(
        d["home_draw_rate_20"] if "home_draw_rate_20" in d.columns else pd.Series(np.nan, index=d.index),
        errors="coerce",
    )
    away_draw = pd.to_numeric(
        d["away_draw_rate_20"] if "away_draw_rate_20" in d.columns else pd.Series(np.nan, index=d.index),
        errors="coerce",
    )
    draw_env = (home_draw + away_draw) / 2.0
    d["routing_draw_environment"] = pd.cut(
        draw_env,
        bins=[-np.inf, 0.22, 0.28, 0.34, np.inf],
        labels=["LOW", "MID_LOW", "MID_HIGH", "HIGH"],
    ).astype("string").fillna("MISSING")

    rest_source = (
        d["rest_diff_hours"]
        if "rest_diff_hours" in d.columns
        else pd.Series(np.nan, index=d.index, dtype=float)
    )
    rest = pd.to_numeric(rest_source, errors="coerce")
    d["routing_rest"] = pd.cut(
        rest,
        bins=[-np.inf, -24.0, -6.0, 6.0, 24.0, np.inf],
        labels=["AWAY_MAJOR", "AWAY_SMALL", "EVEN", "HOME_SMALL", "HOME_MAJOR"],
    ).astype("string").fillna("MISSING")

    if "neutral_venue_known" in d.columns:
        known = d["neutral_venue_known"].astype("boolean")
        neutral = d.get("neutral_venue", pd.Series(False, index=d.index)).astype("boolean")
        d["routing_venue"] = np.where(
            known.fillna(False),
            np.where(neutral.fillna(False), "NEUTRAL", "HOME_AWAY"),
            "UNKNOWN",
        )
    else:
        d["routing_venue"] = "UNKNOWN"

    d["routing_context"] = (
        d.get("competition", pd.Series("__MISSING__", index=d.index)).astype("string").fillna("__MISSING__")
        + "|"
        + d["routing_strength_gap"].fillna("MISSING")
        + "|"
        + d["routing_scoring_environment"].fillna("MISSING")
        + "|"
        + d["routing_draw_environment"].fillna("MISSING")
        + "|"
        + d["routing_rest"].fillna("MISSING")
        + "|"
        + d["routing_venue"].astype("string")
    )
    return d



def _context_keys(frame: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """Return routing keys from most specific to broadest granularity."""
    d = frame
    levels: list[tuple[str, pd.Series]] = []
    if "routing_context" in d.columns:
        levels.append(("FULL", d["routing_context"].astype("string")))
    if {"competition", "routing_strength_gap"}.issubset(d.columns):
        levels.append((
            "COMP_STRENGTH",
            d["competition"].astype("string").fillna("__MISSING__")
            + "|"
            + d["routing_strength_gap"].astype("string").fillna("MISSING"),
        ))
    if "competition" in d.columns:
        levels.append(("COMP", d["competition"].astype("string").fillna("__MISSING__")))
    return levels


def _lookup_context_weights(
    row: pd.Series,
    context_weights: dict[str, dict[str, float]],
    fallback: dict[str, float],
) -> tuple[dict[str, float], str]:
    """Resolve the most specific learned context, then safely fall back."""
    if "routing_context" in row.index:
        full = str(row.get("routing_context", ""))
        comp = str(row.get("competition", "__MISSING__"))
        strength = str(row.get("routing_strength_gap", "MISSING"))
    else:
        routed = _routing_context(pd.DataFrame([row])).iloc[0]
        full = str(routed.get("routing_context", ""))
        comp = str(routed.get("competition", "__MISSING__"))
        strength = str(routed.get("routing_strength_gap", "MISSING"))
    keys = [
        ("FULL", full),
        ("COMP_STRENGTH", f"{comp}|{strength}"),
        ("COMP", comp),
    ]
    for level, key in keys:
        learned = context_weights.get(f"{level}:{key}")
        if learned is not None:
            return learned, f"{level}:{key}"
    return fallback, "GLOBAL"


def _contextual_blend_weights(
    validation: pd.DataFrame,
    validation_models: dict,
    feature_cols: list[str],
    global_weights: dict[str, float],
    *,
    min_rows: int = 60,
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """Learn context-specific mixtures with hierarchical sparse-data fallback."""
    routed: dict[str, dict[str, float]] = {}
    reasons: dict[str, str] = {}
    context_data = _routing_context(validation.reset_index(drop=True)).reset_index(drop=True)
    # Each candidate is predicted once over the entire selection slice. Context
    # optimization then reuses those predictions instead of re-scoring every group.
    all_probs = {
        name: np.asarray(model.predict_proba(context_data[feature_cols]), dtype=float)
        for name, model in validation_models.items()
    }

    levels = _context_keys(context_data)
    for level, keys in levels:
        temp = context_data.copy()
        temp["_routing_key"] = keys.astype("string")
        for key, sl in temp.groupby("_routing_key", sort=True, dropna=False):
            context_name = f"{level}:{key}"
            if len(sl) < min_rows:
                routed[context_name] = dict(global_weights)
                reasons[context_name] = "fallback_global_insufficient_validation_rows"
                continue
            positions = sl.index.to_numpy(dtype=int)
            probs = {name: values[positions] for name, values in all_probs.items()}
            optimized, used = _optimize_blend_weights(
                sl.target.astype(int),
                probs,
                global_weights,
            )
            routed[context_name] = optimized if used else dict(global_weights)
            reasons[context_name] = (
                "context_specific_optimized_validation"
                if used
                else "fallback_global_optimizer_failed"
            )

    routed["GLOBAL"] = dict(global_weights)
    reasons["GLOBAL"] = "global_fallback"
    return routed, reasons


def _routed_ensemble_proba(
    frame: pd.DataFrame,
    fitted_models: dict,
    feature_cols: list[str],
    context_weights: dict[str, dict[str, float]],
    fallback: dict[str, float],
) -> tuple[np.ndarray, list[str]]:
    """Batch-predict every model once, then apply hierarchical routing row-wise."""
    if frame.empty:
        return np.empty((0, 3), dtype=float), []

    routed = _routing_context(frame.reset_index(drop=True)).reset_index(drop=True)
    names = list(fitted_models)
    n = len(routed)
    weight_matrix = np.tile(
        np.asarray([float(fallback.get(name, 0.0)) for name in names], dtype=float),
        (n, 1),
    )
    routes = ["GLOBAL"] * n

    for i in range(n):
        comp = str(routed.loc[i, "competition"]) if "competition" in routed.columns else "__MISSING__"
        strength = str(routed.loc[i, "routing_strength_gap"])
        full = str(routed.loc[i, "routing_context"])
        candidates_for_row = (
            ("FULL", full),
            ("COMP_STRENGTH", f"{comp}|{strength}"),
            ("COMP", comp),
        )
        for level, key in candidates_for_row:
            learned = context_weights.get(f"{level}:{key}")
            if learned is not None:
                weight_matrix[i] = np.asarray(
                    [float(learned.get(name, 0.0)) for name in names],
                    dtype=float,
                )
                routes[i] = f"{level}:{key}"
                break

    probs = np.zeros((n, 3), dtype=float)
    for j, name in enumerate(names):
        pred = np.asarray(fitted_models[name].predict_proba(routed[feature_cols]), dtype=float)
        if pred.shape != (n, 3) or not np.isfinite(pred).all():
            raise ValueError(f"Model {name!r} produced invalid routed probabilities")
        pred_sum = pred.sum(axis=1, keepdims=True)
        if np.any(pred_sum <= 0):
            raise ValueError(f"Model {name!r} produced a zero probability row")
        pred = pred / pred_sum
        probs += weight_matrix[:, [j]] * pred

    row_sum = probs.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0) or not np.isfinite(row_sum).all():
        raise ValueError("Routed ensemble produced invalid probability rows")
    probs /= row_sum
    return probs, routes


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
        val_probs, _calibration_routes = _routed_ensemble_proba(
            val_calib,
            validation_models,
            feature_cols,
            context_weights,
            weights,
        )
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
        probs, _oos_routes = _routed_ensemble_proba(
            oos,
            fitted,
            feature_cols,
            context_weights,
            weights,
        )
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
