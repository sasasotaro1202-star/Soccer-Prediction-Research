"""Conservative multi-fold stability gate for research candidates.

The gate is intentionally strict: aggregate OOS improvement is not enough.
A candidate must show repeatable improvement across independent chronological
folds and across more than one competition/season. This module never changes
the incumbent model; it only returns a deterministic research status.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import math
from typing import Any, Mapping, Sequence


PROJECT_MIN_IMPROVED_FRACTION = 0.70


def _metric_delta(candidate: Mapping[str, float], baseline: Mapping[str, float], metric: str) -> float:
    return float(candidate[metric]) - float(baseline[metric])


def _parse_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _expand_axis(value: Any) -> set[str]:
    """Normalize scalar, pipe-delimited, or iterable fold coverage metadata."""
    if value is None:
        return set()
    if isinstance(value, str):
        return {part.strip() for part in value.split("|") if part.strip()}
    if isinstance(value, Iterable):
        return {str(part).strip() for part in value if str(part).strip()}
    return {str(value).strip()}


def _coverage(folds: Sequence[Mapping[str, Any]], key: str) -> set[str]:
    values: set[str] = set()
    for fold in folds:
        values.update(_expand_axis(fold.get(key)))
    return values


def _validate_chronology(folds: Sequence[Mapping[str, Any]]) -> tuple[bool, str, dict[str, Any]]:
    """Require explicit, non-overlapping chronological OOS boundaries."""
    previous_end: datetime | None = None
    boundaries: list[dict[str, str]] = []
    for i, fold in enumerate(folds):
        start = _parse_utc(fold.get("oos_start_utc"))
        end = _parse_utc(fold.get("oos_end_utc"))
        if start is None or end is None:
            return False, "chronology_evidence_missing_or_invalid", {"fold": i}
        if end <= start:
            return False, "chronology_boundary_order_invalid", {"fold": i}
        if previous_end is not None and start < previous_end:
            return False, "chronology_folds_overlap_or_reverse", {"fold": i}
        previous_end = end
        boundaries.append({"oos_start_utc": start.isoformat(), "oos_end_utc": end.isoformat()})
    return True, "ok", {"boundaries": boundaries}


def _finite_metrics(folds: Sequence[Mapping[str, Any]]) -> tuple[bool, int | None]:
    required = ("logloss", "brier", "accuracy")
    for i, fold in enumerate(folds):
        base = fold.get("baseline", {})
        cand = fold.get("candidate", {})
        for metric in required:
            try:
                if not math.isfinite(float(base[metric])) or not math.isfinite(float(cand[metric])):
                    return False, i
            except (TypeError, ValueError, KeyError):
                return False, i
    return True, None


def evaluate_stability(
    folds: Sequence[Mapping[str, Any]],
    *,
    min_folds: int = 3,
    min_unique_leagues: int = 2,
    min_unique_seasons: int = 2,
    min_improved_fraction: float = PROJECT_MIN_IMPROVED_FRACTION,
) -> dict[str, Any]:
    """Return HOLD unless chronological-fold improvement is repeatable.

    The project-level adoption benchmark requires non-regression in at least
    70% of development evaluation periods. Callers may pass a stricter or
    explicitly justified threshold, but the default must not be weaker than
    the project benchmark.
    """
    try:
        required_fraction = float(min_improved_fraction)
    except (TypeError, ValueError):
        return {
            "status": "HOLD",
            "reason": "invalid_improved_fraction",
            "min_improved_fraction": min_improved_fraction,
        }
    if not 0.70 <= required_fraction <= 1.0:
        return {
            "status": "HOLD",
            "reason": "improved_fraction_below_project_minimum_or_invalid",
            "min_improved_fraction": required_fraction,
            "project_min_improved_fraction": PROJECT_MIN_IMPROVED_FRACTION,
        }

    if len(folds) < min_folds:
        return {"status": "HOLD", "reason": "too_few_folds", "folds": len(folds)}

    chronology_ok, chronology_reason, chronology = _validate_chronology(folds)
    if not chronology_ok:
        return {
            "status": "HOLD",
            "reason": chronology_reason,
            **chronology,
        }

    leagues = _coverage(folds, "league")
    seasons = _coverage(folds, "season")
    if len(leagues) < min_unique_leagues:
        return {"status": "HOLD", "reason": "too_few_unique_leagues", "unique_leagues": sorted(leagues)}
    if len(seasons) < min_unique_seasons:
        return {"status": "HOLD", "reason": "too_few_unique_seasons", "unique_seasons": sorted(seasons)}

    required = ("logloss", "brier", "accuracy")
    for i, fold in enumerate(folds):
        base = fold.get("baseline", {})
        cand = fold.get("candidate", {})
        if not isinstance(base, Mapping) or not isinstance(cand, Mapping):
            return {"status": "HOLD", "reason": "incomplete_fold_metrics", "fold": i}
        if not all(k in base and k in cand for k in required):
            return {"status": "HOLD", "reason": "incomplete_fold_metrics", "fold": i}

    finite_ok, bad_fold = _finite_metrics(folds)
    if not finite_ok:
        return {"status": "HOLD", "reason": "non_finite_fold_metrics", "fold": bad_fold}

    ll_deltas = [_metric_delta(f["candidate"], f["baseline"], "logloss") for f in folds]
    brier_deltas = [_metric_delta(f["candidate"], f["baseline"], "brier") for f in folds]
    acc_deltas = [_metric_delta(f["candidate"], f["baseline"], "accuracy") for f in folds]

    ll_good = sum(d <= 0 for d in ll_deltas)
    brier_good = sum(d <= 0 for d in brier_deltas)
    acc_good = sum(d >= 0 for d in acc_deltas)
    n = len(folds)

    stable = (
        ll_good / n >= required_fraction
        and brier_good / n >= required_fraction
        and acc_good / n >= required_fraction
        and max(ll_deltas) <= 0
    )
    return {
        "status": "PASS" if stable else "HOLD",
        "folds": n,
        "unique_leagues": sorted(leagues),
        "unique_seasons": sorted(seasons),
        "chronology_verified": True,
        **chronology,
        "logloss_improved_folds": ll_good,
        "brier_improved_or_equal_folds": brier_good,
        "accuracy_improved_or_equal_folds": acc_good,
        "worst_logloss_delta": max(ll_deltas),
        "worst_brier_delta": max(brier_deltas),
        "worst_accuracy_delta": min(acc_deltas),
        "min_improved_fraction": required_fraction,
        "project_min_improved_fraction": PROJECT_MIN_IMPROVED_FRACTION,
        "finite_metrics_verified": True,
    }
