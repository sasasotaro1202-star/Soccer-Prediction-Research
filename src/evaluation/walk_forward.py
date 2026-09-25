from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from src.evaluation.metrics import classification_metrics
from src.models.baselines import candidates
from src.monitoring.dynamic_routing import build_drift_reference, compute_drift_scores, dynamic_route_weights, history_support_risk, routing_risk_bucket, routing_risk_score

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


def _contextual_temperatures(
    y: pd.Series,
    proba: np.ndarray,
    routes: list[str],
    global_temperature: float,
    *,
    min_rows: int = 60,
    prior_strength: int = 240,
) -> tuple[dict[str, float], dict[str, str]]:
    """Fit sparse-safe route temperatures using calibration rows only."""
    if len(y) != len(routes) or np.asarray(proba).shape != (len(y), 3):
        return {"GLOBAL": float(global_temperature)}, {"GLOBAL": "global_fallback_invalid_input"}

    routed = pd.Series(routes, dtype="string").fillna("GLOBAL")
    temperatures: dict[str, float] = {"GLOBAL": float(global_temperature)}
    reasons: dict[str, str] = {"GLOBAL": "global_fallback"}

    yv = np.asarray(y, dtype=int)
    pv = np.asarray(proba, dtype=float)
    for route, positions in routed.groupby(routed, sort=True).groups.items():
        route_key = str(route)
        idx = np.asarray(list(positions), dtype=int)
        if route_key == "GLOBAL":
            temperatures[route_key] = float(global_temperature)
            reasons[route_key] = "global_fallback"
            continue
        if len(idx) < int(min_rows):
            temperatures[route_key] = float(global_temperature)
            reasons[route_key] = "fallback_global_insufficient_calibration_rows"
            continue

        fitted, used = _fit_temperature(yv[idx], pv[idx])
        if not used or not np.isfinite(fitted):
            temperatures[route_key] = float(global_temperature)
            reasons[route_key] = "fallback_global_context_temperature_not_improved"
            continue

        # Empirical-Bayes-style shrinkage limits overreaction in sparse contexts.
        alpha = float(len(idx) / (len(idx) + max(int(prior_strength), 1)))
        shrunk = float(global_temperature + alpha * (fitted - global_temperature))
        temperatures[route_key] = float(np.clip(shrunk, 0.70, 1.60))
        reasons[route_key] = "context_temperature_shrunk_from_calibration"

    return temperatures, reasons


def _risk_temperature_modifiers(
    y: pd.Series,
    proba: np.ndarray,
    risk_scores: np.ndarray,
    *,
    min_rows: int = 80,
    prior_strength: int = 240,
) -> tuple[dict[str, float], dict[str, str]]:
    """Fit small risk-bucket temperature modifiers on calibration rows only."""
    if len(y) != len(proba) or len(y) != len(risk_scores):
        raise ValueError("Risk calibration inputs must have equal lengths")
    raw = np.asarray(proba, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
        raise ValueError("Risk calibration probabilities are invalid")
    buckets = routing_risk_bucket(np.asarray(risk_scores, dtype=float))
    yv = np.asarray(y, dtype=int)
    modifiers: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for bucket in ("LOW", "MEDIUM", "HIGH"):
        idx = np.flatnonzero(buckets == bucket)
        if len(idx) < int(min_rows):
            modifiers[bucket] = 1.0
            reasons[bucket] = "identity_sparse_risk_bucket"
            continue
        fitted, improved = _fit_temperature(
            pd.Series(yv[idx]),
            raw[idx],
        )
        alpha = float(len(idx) / (len(idx) + max(int(prior_strength), 1)))
        modifier = 1.0 + alpha * (float(fitted) - 1.0)
        modifier = float(np.clip(modifier, 0.85, 1.20))
        if not improved:
            modifier = 1.0
            reasons[bucket] = "identity_no_calibration_gain"
        else:
            reasons[bucket] = "risk_bucket_temperature_shrunk"
        modifiers[bucket] = modifier
    return modifiers, reasons


def _apply_risk_temperature_modifiers(
    proba: np.ndarray,
    risk_scores: np.ndarray,
    modifiers: dict[str, float],
) -> np.ndarray:
    """Apply bounded multiplicative temperature corrections by prediction-time risk."""
    p = np.asarray(proba, dtype=float)
    scores = np.asarray(risk_scores, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) != len(scores):
        raise ValueError("Risk temperature probability/score shape mismatch")
    buckets = routing_risk_bucket(scores)
    out = np.empty_like(p, dtype=float)
    for i, bucket in enumerate(buckets):
        temperature = float(modifiers.get(str(bucket), 1.0))
        if not np.isfinite(temperature) or not 0.85 <= temperature <= 1.20:
            raise ValueError("Risk temperature modifier is outside the production bound")
        out[i] = _temperature_transform(p[i:i + 1], temperature)[0]
    if not np.isfinite(out).all():
        raise ValueError("Risk temperature calibration produced invalid probabilities")
    return out


def _apply_contextual_temperatures(
    proba: np.ndarray,
    routes: list[str],
    temperatures: dict[str, float],
    fallback: float,
) -> np.ndarray:
    p = np.asarray(proba, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(routes) != len(p):
        raise ValueError("Probability/route shape mismatch in contextual temperature calibration")
    out = np.empty_like(p, dtype=float)
    for i, route in enumerate(routes):
        temperature = float(temperatures.get(str(route), fallback))
        if not np.isfinite(temperature) or temperature <= 0:
            temperature = float(fallback)
        out[i] = _temperature_transform(p[i : i + 1], temperature)[0]
    if not np.isfinite(out).all():
        raise ValueError("Contextual temperature calibration produced invalid probabilities")
    return out


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
    if {"competition", "routing_strength_gap", "routing_draw_environment"}.issubset(d.columns):
        levels.append((
            "COMP_STRENGTH_DRAW",
            d["competition"].astype("string").fillna("__MISSING__")
            + "|"
            + d["routing_strength_gap"].astype("string").fillna("MISSING")
            + "|"
            + d["routing_draw_environment"].astype("string").fillna("MISSING"),
        ))
    if {"competition", "routing_draw_environment"}.issubset(d.columns):
        levels.append((
            "COMP_DRAW",
            d["competition"].astype("string").fillna("__MISSING__")
            + "|"
            + d["routing_draw_environment"].astype("string").fillna("MISSING"),
        ))
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
        draw_env = str(row.get("routing_draw_environment", "MISSING"))
    else:
        routed = _routing_context(pd.DataFrame([row])).iloc[0]
        full = str(routed.get("routing_context", ""))
        comp = str(routed.get("competition", "__MISSING__"))
        strength = str(routed.get("routing_strength_gap", "MISSING"))
        draw_env = str(routed.get("routing_draw_environment", "MISSING"))
    keys = [
        ("FULL", full),
        ("COMP_STRENGTH_DRAW", f"{comp}|{strength}|{draw_env}"),
        ("COMP_DRAW", f"{comp}|{draw_env}"),
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
    prior_strength: int = 240,
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
            if not used:
                routed[context_name] = dict(global_weights)
                reasons[context_name] = "fallback_global_optimizer_failed"
                continue

            # Empirical-Bayes-style shrinkage keeps sparse context weights close
            # to the globally validated mixture.
            alpha = float(len(sl) / (len(sl) + max(int(prior_strength), 1)))
            names = list(global_weights)
            local = np.asarray([float(optimized.get(name, 0.0)) for name in names], dtype=float)
            anchor = np.asarray([float(global_weights.get(name, 0.0)) for name in names], dtype=float)
            if not np.isfinite(local).all() or not np.isfinite(anchor).all():
                routed[context_name] = dict(global_weights)
                reasons[context_name] = "fallback_global_invalid_optimized_weights"
                continue
            shrunk = anchor + alpha * (local - anchor)
            shrunk = np.clip(shrunk, 0.0, None)
            total = float(shrunk.sum())
            if total <= 0.0 or not np.isfinite(total):
                routed[context_name] = dict(global_weights)
                reasons[context_name] = "fallback_global_invalid_shrunk_weights"
                continue
            shrunk /= total
            routed[context_name] = {name: float(value) for name, value in zip(names, shrunk)}
            reasons[context_name] = "context_specific_optimized_validation_shrunk"

    routed["GLOBAL"] = dict(global_weights)
    reasons["GLOBAL"] = "global_fallback"
    return routed, reasons


def _routed_ensemble_proba(
    frame: pd.DataFrame,
    fitted_models: dict,
    feature_cols: list[str],
    context_weights: dict[str, dict[str, float]],
    fallback: dict[str, float],
    dynamic_policy: dict | None = None,
    *,
    return_diagnostics: bool = False,
) -> tuple[np.ndarray, list[str]] | tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    """Batch-predict every model once, then apply hierarchical routing row-wise."""
    if frame.empty:
        empty_diag = {
            "drift": np.empty(0, dtype=float),
            "uncertainty": np.empty(0, dtype=float),
            "entropy": np.empty(0, dtype=float),
            "disagreement": np.empty(0, dtype=float),
            "trust": np.empty(0, dtype=float),
            "risk": np.empty(0, dtype=float),
        }
        return (np.empty((0, 3), dtype=float), [], empty_diag) if return_diagnostics else (np.empty((0, 3), dtype=float), [])

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
        draw_env = str(routed.loc[i, "routing_draw_environment"])
        candidates_for_row = (
            ("FULL", full),
            ("COMP_STRENGTH_DRAW", f"{comp}|{strength}|{draw_env}"),
            ("COMP_DRAW", f"{comp}|{draw_env}"),
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

    model_predictions: dict[str, np.ndarray] = {}
    for name in names:
        pred = np.asarray(fitted_models[name].predict_proba(routed[feature_cols]), dtype=float)
        if pred.shape != (n, 3) or not np.isfinite(pred).all():
            raise ValueError(f"Model {name!r} produced invalid routed probabilities")
        pred_sum = pred.sum(axis=1, keepdims=True)
        if np.any(pred_sum <= 0):
            raise ValueError(f"Model {name!r} produced a zero probability row")
        model_predictions[name] = pred / pred_sum

    effective_weights = weight_matrix
    routing_diag: dict[str, np.ndarray] = {
        "drift": np.zeros(n, dtype=float),
        "uncertainty": np.zeros(n, dtype=float),
        "entropy": np.zeros(n, dtype=float),
        "disagreement": np.zeros(n, dtype=float),
        "trust": np.ones(n, dtype=float),
        "risk": np.zeros(n, dtype=float),
    }
    if isinstance(dynamic_policy, dict) and bool(dynamic_policy.get("enabled", False)):
        reference = dynamic_policy.get("reference")
        dynamic_feature_cols = dynamic_policy.get("feature_cols") or feature_cols
        drift_scores = compute_drift_scores(routed, reference or {}, list(dynamic_feature_cols))
        support_scores = history_support_risk(routed, min_games=5.0, full_games=20.0)
        fallback_source = dynamic_policy.get("fallback_weights", fallback)
        fallback_vector = np.asarray([float(fallback_source.get(name, 0.0)) for name in names], dtype=float)
        effective_weights, diag = dynamic_route_weights(
            weight_matrix,
            fallback_vector,
            model_predictions,
            drift_scores,
            drift_strength=float(dynamic_policy.get("drift_strength", 0.85)),
            uncertainty_strength=float(dynamic_policy.get("uncertainty_strength", 0.75)),
            min_specialist_trust=float(dynamic_policy.get("min_specialist_trust", 0.25)),
            support_scores=support_scores,
            support_strength=float(dynamic_policy.get("support_strength", 0.40)),
        )
        routing_diag.update({str(k): np.asarray(v, dtype=float) for k, v in diag.items()})
        routing_diag["risk"] = routing_risk_score(
            routing_diag["drift"],
            routing_diag["uncertainty"],
            support_scores=routing_diag.get("support"),
            support_weight=float(dynamic_policy.get("support_risk_weight", 0.0)),
        )

    probs = np.zeros((n, 3), dtype=float)
    for j, name in enumerate(names):
        probs += effective_weights[:, [j]] * model_predictions[name]

    row_sum = probs.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0) or not np.isfinite(row_sum).all():
        raise ValueError("Routed ensemble produced invalid probability rows")
    probs /= row_sum
    if return_diagnostics:
        return probs, routes, routing_diag
    return probs, routes


def _advance_past_same_kickoff(df: pd.DataFrame, index: int) -> int:
    """Move a row-count boundary past all rows sharing the same kickoff time."""
    boundary = int(index)
    if boundary <= 0 or boundary >= len(df):
        return boundary
    kickoff = df.iloc[boundary - 1]["kickoff_utc"]
    while boundary < len(df) and df.iloc[boundary]["kickoff_utc"] == kickoff:
        boundary += 1
    return boundary


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
    start = _advance_past_same_kickoff(d, min_train)
    while start < len(d):
        oos_end = _advance_past_same_kickoff(d, min(start + oos_block, len(d)))
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
        dynamic_policy = {
            "schema_version": 1,
            "type": "drift_uncertainty_router",
            "enabled": True,
            "drift_strength": 0.85,
            "uncertainty_strength": 0.75,
            "min_specialist_trust": 0.25,
            "support_strength": 0.40,
            "support_risk_weight": 0.15,
            "support_cols": ["home_history_support_n", "away_history_support_n"],
            "feature_cols": list(feature_cols),
            "reference": build_drift_reference(fit, feature_cols),
        }
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
        val_probs, _calibration_routes, calibration_risk_diag = _routed_ensemble_proba(
            val_calib,
            validation_models,
            feature_cols,
            context_weights,
            weights,
            dynamic_policy=dynamic_policy,
            return_diagnostics=True,
        )
        val_probs = np.clip(val_probs, 1e-9, 1.0)
        val_probs /= val_probs.sum(axis=1, keepdims=True)
        calibration_y = val_calib.target.astype(int)
        calibration_temperature, calibration_used = _fit_temperature(calibration_y, val_probs)
        contextual_temperatures, contextual_temperature_reasons = _contextual_temperatures(
            calibration_y,
            val_probs,
            _calibration_routes,
            calibration_temperature,
        )
        # Risk calibration is layered on top of the context calibration and is
        # deliberately low-amplitude so uncertainty never becomes a large ad-hoc shift.
        context_calibrated = _apply_contextual_temperatures(
            val_probs,
            _calibration_routes,
            contextual_temperatures,
            calibration_temperature,
        )
        risk_temperature_modifiers, risk_temperature_reasons = _risk_temperature_modifiers(
            calibration_y,
            context_calibrated,
            calibration_risk_diag["risk"],
        )

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
            "contextual_temperatures": contextual_temperatures,
            "contextual_temperature_reasons": contextual_temperature_reasons,
            "risk_temperature_modifiers": risk_temperature_modifiers,
            "risk_temperature_reasons": risk_temperature_reasons,
            "dynamic_routing": dynamic_policy,
        })

        # Refit candidates on all historical data available before this OOS block.
        fitted = {
            name: _fit_predict(model, train[feature_cols], train.target.astype(int))
            for name, model in candidates(random_state).items()
        }
        probs, _oos_routes, oos_risk_diag = _routed_ensemble_proba(
            oos,
            fitted,
            feature_cols,
            context_weights,
            weights,
            dynamic_policy=dynamic_policy,
            return_diagnostics=True,
        )
        probs = np.clip(probs, 1e-9, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
        probs = _apply_contextual_temperatures(
            probs,
            _oos_routes,
            contextual_temperatures,
            calibration_temperature,
        )
        probs = _apply_risk_temperature_modifiers(
            probs,
            oos_risk_diag["risk"],
            risk_temperature_modifiers,
        )

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
