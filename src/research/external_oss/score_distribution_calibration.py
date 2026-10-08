"""Research-only temperature calibration for joint soccer score distributions."""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


def _normalize(dist: Any) -> list[tuple[int, int, float]]:
    rows = [(int(h), int(a), float(p)) for h, a, p in dist]
    if not rows:
        raise ValueError("empty score distribution")
    probs = np.asarray([p for _, _, p in rows], dtype=float)
    if not np.isfinite(probs).all() or (probs < 0).any() or probs.sum() <= 0:
        raise ValueError("invalid score probabilities")
    probs /= probs.sum()
    return [(h, a, float(p)) for (h, a, _), p in zip(rows, probs)]


def temperature_transform(
    dist: Any,
    temperature: float,
) -> list[tuple[int, int, float]]:
    """Apply p(score)^(1/T) and renormalize."""
    t = float(temperature)
    if not np.isfinite(t) or t <= 0:
        raise ValueError("temperature must be finite and positive")
    normalized = _normalize(dist)
    logp = np.log(
        np.clip(np.asarray([p for _, _, p in normalized], dtype=float), 1e-15, 1.0)
    ) / t
    logp -= float(np.max(logp))
    probs = np.exp(logp)
    probs /= float(probs.sum())
    return [(h, a, float(p)) for (h, a, _), p in zip(normalized, probs)]


def _build_calibration_arrays(
    calibration_frame: pd.DataFrame,
    model: Any,
    distribution_fn: Callable[..., Any],
) -> tuple[np.ndarray, np.ndarray]:
    log_probs: list[np.ndarray] = []
    actual_indices: list[int] = []

    for row in calibration_frame.itertuples(index=False):
        dist = _normalize(
            distribution_fn(
                model,
                row.home_team,
                row.away_team,
                row.competition,
                max_goals=12,
            )
        )
        states = [(h, a) for h, a, _ in dist]
        actual = (int(row.home_goals), int(row.away_goals))
        if actual not in states:
            raise ValueError(
                "calibration outcome is outside the model score support; "
                "increase max_goals or reject the fold"
            )
        probs = np.asarray([p for _, _, p in dist], dtype=float)
        log_probs.append(np.log(np.clip(probs, 1e-15, 1.0)))
        actual_indices.append(states.index(actual))

    if not log_probs:
        raise ValueError("calibration frame is empty")
    if len({len(x) for x in log_probs}) != 1:
        raise ValueError("calibration distributions have inconsistent support")
    return np.vstack(log_probs), np.asarray(actual_indices, dtype=int)


def fit_temperature(
    calibration_frame: pd.DataFrame,
    model: Any,
    distribution_fn: Callable[..., Any],
    *,
    lower: float = 0.25,
    upper: float = 4.0,
) -> dict[str, Any]:
    """Fit one scalar temperature using only a chronological calibration slice."""
    required = {"home_team", "away_team", "competition", "home_goals", "away_goals"}
    missing = sorted(required - set(calibration_frame.columns))
    if missing:
        raise ValueError(f"calibration data missing columns: {missing}")
    if calibration_frame.empty:
        raise ValueError("calibration frame is empty")
    if not (0 < float(lower) < float(upper)):
        raise ValueError("invalid temperature bounds")

    log_probs, actual_indices = _build_calibration_arrays(
        calibration_frame, model, distribution_fn
    )
    row_index = np.arange(len(actual_indices))

    def objective(log_temperature: float) -> float:
        temperature = math.exp(float(log_temperature))
        scaled = log_probs / temperature
        log_normalizer = logsumexp(scaled, axis=1)
        nll = -(scaled[row_index, actual_indices] - log_normalizer)
        return float(np.mean(nll))

    raw_nll = objective(0.0)
    result = minimize_scalar(
        objective,
        bounds=(math.log(float(lower)), math.log(float(upper))),
        method="bounded",
        options={"xatol": 1e-3, "maxiter": 80},
    )
    if not result.success or not np.isfinite(float(result.fun)):
        raise RuntimeError("temperature optimization failed")

    temperature = float(math.exp(float(result.x)))
    return {
        "temperature": temperature,
        "calibration_rows": int(len(calibration_frame)),
        "raw_calibration_logloss": float(raw_nll),
        "calibrated_calibration_logloss": float(result.fun),
        "optimization_success": bool(result.success),
        "method": "joint_score_temperature_scaling",
    }
