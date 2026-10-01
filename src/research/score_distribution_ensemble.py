"""Research-only distribution-level ensemble for soccer score models.

Each candidate produces a joint P(home_goals, away_goals) distribution. The
ensemble mixes those full distributions, then derives exact score, O/U 2.5,
BTTS, and goal-count outputs from the same joint object.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd


def normalize_distribution(dist: Any) -> dict[tuple[int, int], float]:
    values: dict[tuple[int, int], float] = {}
    for item in dist:
        if len(item) < 3:
            raise ValueError("score distribution row must contain home, away, probability")
        h, a, prob = int(item[0]), int(item[1]), float(item[2])
        if h < 0 or a < 0 or not np.isfinite(prob) or prob < 0:
            raise ValueError("invalid score distribution value")
        values[(h, a)] = values.get((h, a), 0.0) + prob
    if not values:
        raise ValueError("empty score distribution")
    total = float(sum(values.values()))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("score distribution has invalid total mass")
    return {key: value / total for key, value in values.items()}


def blend_distributions(
    distributions: Mapping[str, Any],
    weights: Mapping[str, float],
) -> dict[tuple[int, int], float]:
    names = [name for name in distributions if name in weights]
    if not names:
        raise ValueError("no overlapping distributions and weights")
    raw_weights = np.asarray([float(weights[name]) for name in names], dtype=float)
    if not np.isfinite(raw_weights).all() or (raw_weights < 0).any() or raw_weights.sum() <= 0:
        raise ValueError("invalid ensemble weights")
    raw_weights /= raw_weights.sum()

    normalized = {name: normalize_distribution(distributions[name]) for name in names}
    support = sorted(set().union(*(set(d) for d in normalized.values())))
    out = {}
    for key in support:
        out[key] = float(
            sum(raw_weights[i] * normalized[name].get(key, 0.0) for i, name in enumerate(names))
        )
    total = float(sum(out.values()))
    if total <= 0 or not np.isfinite(total):
        raise ValueError("blended distribution has invalid mass")
    return {key: value / total for key, value in out.items()}


def derive_metrics(
    dist: Mapping[tuple[int, int], float],
    actual_home: int,
    actual_away: int,
) -> dict[str, float]:
    d = normalize_distribution([(h, a, p) for (h, a), p in dist.items()])
    ranked = sorted(d.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
    actual = (int(actual_home), int(actual_away))
    actual_prob = float(d.get(actual, 0.0))
    top3 = {key for key, _ in ranked[:3]}
    top4 = {key for key, _ in ranked[:4]}
    best_h, best_a = ranked[0][0]
    p_over25 = float(sum(p for (h, a), p in d.items() if h + a >= 3))
    p_btts = float(sum(p for (h, a), p in d.items() if h >= 1 and a >= 1))
    y_over25 = int(actual_home + actual_away >= 3)
    y_btts = int(actual_home >= 1 and actual_away >= 1)

    def binary_logloss(y: int, p: float) -> float:
        q = float(np.clip(p, 1e-12, 1.0 - 1e-12))
        return float(-(y * math.log(q) + (1 - y) * math.log(1 - q)))

    return {
        "score_logloss": float(-math.log(max(actual_prob, 1e-12))),
        "exact_score_hit": float(actual == ranked[0][0]),
        "top3_score_hit": float(actual in top3),
        "top4_score_hit": float(actual in top4),
        "home_goals_abs_error": float(abs(best_h - actual_home)),
        "away_goals_abs_error": float(abs(best_a - actual_away)),
        "total_goals_abs_error": float(abs(best_h + best_a - actual_home - actual_away)),
        "over_2_5_logloss": binary_logloss(y_over25, p_over25),
        "over_2_5_brier": float((p_over25 - y_over25) ** 2),
        "btts_logloss": binary_logloss(y_btts, p_btts),
        "btts_brier": float((p_btts - y_btts) ** 2),
    }


def cross_block_weights(
    history: Mapping[str, list[float]],
    names: list[str],
    *,
    temperature: float = 0.20,
    min_weight: float = 0.05,
    max_weight: float = 0.75,
) -> dict[str, float]:
    if not names:
        return {}
    if min_weight < 0 or max_weight <= 0 or min_weight > max_weight:
        raise ValueError("invalid weight bounds")
    if min_weight * len(names) > 1.0 or max_weight * len(names) < 1.0:
        raise ValueError("weight bounds cannot form a simplex")
    losses = []
    for name in names:
        vals = np.asarray(history.get(name, []), dtype=float)
        vals = vals[np.isfinite(vals)]
        losses.append(float(vals.mean()) if len(vals) else math.log(3.0))
    x = -np.asarray(losses, dtype=float) / max(float(temperature), 1e-6)
    x -= float(np.max(x))
    w = np.exp(x)
    w /= w.sum()
    w = np.clip(w, min_weight, max_weight)
    w /= w.sum()
    return {name: float(value) for name, value in zip(names, w)}


def evaluate_block(
    block: pd.DataFrame,
    candidate_models: Mapping[str, Any],
    candidate_distributions: Mapping[str, Callable[..., Any]],
    history: Mapping[str, list[float]],
) -> tuple[dict[str, float], dict[str, float]]:
    names = [
        name for name in ("primary", "dixon_coles", "xg")
        if name in candidate_models and name in candidate_distributions
    ]
    if len(names) < 2:
        raise ValueError("distribution ensemble needs at least two candidate models")
    weights = cross_block_weights(history, names)
    values: list[dict[str, float]] = []

    for row in block.itertuples(index=False):
        distributions: dict[str, Any] = {}
        neutral = bool(getattr(row, "neutral_venue", False)) if hasattr(row, "neutral_venue") else False
        if isinstance(neutral, float) and np.isnan(neutral):
            neutral = False
        for name in names:
            kwargs = {"max_goals": 12}
            if name == "dixon_coles":
                kwargs = {"max_goals": 12}
            if name == "xg":
                kwargs = {"max_goals": 12}
                distributions[name] = candidate_distributions[name](
                    candidate_models[name],
                    row.home_team,
                    row.away_team,
                    row.competition if hasattr(row, "competition") else None,
                    **kwargs,
                )
            else:
                if name == "primary" and "neutral" in str(candidate_models[name].get("method", "")).lower():
                    kwargs["neutral_venue"] = neutral
                distributions[name] = candidate_distributions[name](
                    candidate_models[name],
                    row.home_team,
                    row.away_team,
                    row.competition if hasattr(row, "competition") else None,
                    **kwargs,
                )
        blended = blend_distributions(distributions, weights)
        values.append(derive_metrics(blended, int(row.home_goals), int(row.away_goals)))

    frame = pd.DataFrame(values)
    metrics = {key: float(frame[key].mean()) for key in frame.columns}
    return metrics, weights
