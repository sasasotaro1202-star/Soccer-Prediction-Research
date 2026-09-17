"""Conservative multi-fold stability gate for research candidates.

Inspired by temporal, calibration-first forecasting projects: aggregate OOS
improvement is not enough. A candidate must show repeatable improvement across
independent chronological folds and across more than one competition/season.
This module never changes the incumbent model; it only returns a deterministic
research status.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def _metric_delta(candidate: Mapping[str, float], baseline: Mapping[str, float], metric: str) -> float:
    return float(candidate[metric]) - float(baseline[metric])


def evaluate_stability(
    folds: Sequence[Mapping[str, Any]],
    *,
    min_folds: int = 3,
    min_unique_leagues: int = 2,
    min_unique_seasons: int = 2,
    min_improved_fraction: float = 2 / 3,
) -> dict[str, Any]:
    """Return HOLD unless chronological-fold improvement is repeatable."""
    if len(folds) < min_folds:
        return {"status": "HOLD", "reason": "too_few_folds", "folds": len(folds)}

    leagues = {str(f.get("league")) for f in folds if f.get("league") is not None}
    seasons = {str(f.get("season")) for f in folds if f.get("season") is not None}
    if len(leagues) < min_unique_leagues:
        return {"status": "HOLD", "reason": "too_few_unique_leagues", "unique_leagues": sorted(leagues)}
    if len(seasons) < min_unique_seasons:
        return {"status": "HOLD", "reason": "too_few_unique_seasons", "unique_seasons": sorted(seasons)}

    required = ("logloss", "brier", "accuracy")
    for i, fold in enumerate(folds):
        base = fold.get("baseline", {})
        cand = fold.get("candidate", {})
        if not all(k in base and k in cand for k in required):
            return {"status": "HOLD", "reason": "incomplete_fold_metrics", "fold": i}

    ll_deltas = [_metric_delta(f["candidate"], f["baseline"], "logloss") for f in folds]
    brier_deltas = [_metric_delta(f["candidate"], f["baseline"], "brier") for f in folds]
    acc_deltas = [_metric_delta(f["candidate"], f["baseline"], "accuracy") for f in folds]

    ll_good = sum(d <= 0 for d in ll_deltas)
    brier_good = sum(d <= 0 for d in brier_deltas)
    acc_good = sum(d >= 0 for d in acc_deltas)
    n = len(folds)

    stable = (
        ll_good / n >= min_improved_fraction
        and brier_good / n >= min_improved_fraction
        and acc_good / n >= min_improved_fraction
        and max(ll_deltas) <= 0
    )
    return {
        "status": "PASS" if stable else "HOLD",
        "folds": n,
        "unique_leagues": sorted(leagues),
        "unique_seasons": sorted(seasons),
        "logloss_improved_folds": ll_good,
        "brier_improved_or_equal_folds": brier_good,
        "accuracy_improved_or_equal_folds": acc_good,
        "worst_logloss_delta": max(ll_deltas),
        "worst_brier_delta": max(brier_deltas),
        "worst_accuracy_delta": min(acc_deltas),
        "min_improved_fraction": min_improved_fraction,
    }
