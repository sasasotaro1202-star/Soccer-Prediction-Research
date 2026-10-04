"""Independent-OOS adoption gate for legacy-vs-candidate promotion."""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Mapping, Sequence

from src.research.stability_gate import evaluate_stability


MIN_PRIMARY_RELATIVE_IMPROVEMENT = 0.03
MIN_AUXILIARY_RELATIVE_IMPROVEMENT = 0.01
MAX_ECE_REGRESSION = 0.0


def _better(candidate: Mapping[str, float], baseline: Mapping[str, float], key: str, lower: bool) -> bool:
    return float(candidate[key]) < float(baseline[key]) if lower else float(candidate[key]) > float(baseline[key])


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


def _holdout_integrity(holdout: Mapping[str, Any]) -> tuple[bool, str]:
    """Fail closed unless the holdout is explicitly locked and selection-independent."""
    if holdout.get("locked") is not True:
        return False, "holdout_not_explicitly_locked"
    if holdout.get("selection_frozen") is not True:
        return False, "holdout_selection_not_frozen"
    if holdout.get("used_for_selection") is True:
        return False, "holdout_was_used_for_selection"
    if holdout.get("used_for_calibration") is True:
        return False, "holdout_was_used_for_calibration"
    if holdout.get("used_for_threshold_tuning") is True:
        return False, "holdout_was_used_for_threshold_tuning"

    development_end = _parse_utc(holdout.get("development_end_utc"))
    holdout_start = _parse_utc(holdout.get("holdout_start_utc"))
    if development_end is None or holdout_start is None:
        return False, "holdout_temporal_boundaries_missing_or_invalid"
    if holdout_start <= development_end:
        return False, "holdout_overlaps_development_period"
    return True, "ok"


def independent_adoption_gate(
    development: Mapping[str, Any],
    holdout: Mapping[str, Any],
    *,
    min_holdout_rows: int = 100,
    stability_folds: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fail-closed independent-OOS adoption gate.

    A candidate is adoptable only when the holdout is explicitly locked and
    selection-independent, PIT violations are explicitly zero, the same-OOS
    baseline comparison shows at least the project reference improvement, and
    chronological stability evidence passes across multiple folds/leagues/seasons.
    """
    integrity_ok, integrity_reason = _holdout_integrity(holdout)
    if not integrity_ok:
        return {
            "status": "HOLD",
            "reason": integrity_reason,
            "oos_claimed": False,
            "promotion_authority": "deterministic_research_engine",
        }

    if not development:
        return {
            "status": "HOLD",
            "reason": "development_evidence_missing",
            "oos_claimed": False,
            "promotion_authority": "deterministic_research_engine",
        }

    pit_violations = holdout.get("pit_violations")
    if not isinstance(pit_violations, int) or isinstance(pit_violations, bool) or pit_violations < 0:
        return {
            "status": "HOLD",
            "reason": "pit_violation_count_missing_or_invalid",
            "oos_claimed": False,
            "promotion_authority": "deterministic_research_engine",
        }
    if pit_violations != 0:
        return {
            "status": "HOLD",
            "reason": "pit_violations_present",
            "pit_violations": pit_violations,
            "oos_claimed": False,
            "promotion_authority": "deterministic_research_engine",
        }
    pit_status = str(holdout.get("pit_status", "")).upper()
    if pit_status != "PASS":
        return {
            "status": "HOLD",
            "reason": "pit_status_not_pass",
            "pit_status": pit_status or None,
            "oos_claimed": False,
            "promotion_authority": "deterministic_research_engine",
        }

    holdout_rows = holdout.get("n")
    if isinstance(holdout_rows, bool) or not isinstance(holdout_rows, int) or holdout_rows < 0:
        return {
            "status": "HOLD",
            "reason": "holdout_row_count_missing_or_invalid",
            "oos_claimed": False,
            "promotion_authority": "deterministic_research_engine",
        }
    if holdout_rows < min_holdout_rows:
        return {"status": "HOLD", "reason": "independent_holdout_too_small", "oos_claimed": False}
    if holdout.get("same_oos") is not True:
        return {"status": "HOLD", "reason": "holdout_is_not_same_oos", "oos_claimed": False}

    if stability_folds is None:
        return {
            "status": "HOLD",
            "reason": "stability_evidence_missing",
            "oos_claimed": False,
            "promotion_authority": "deterministic_research_engine",
        }
    stability_result = evaluate_stability(stability_folds)
    if stability_result.get("status") != "PASS":
        return {
            "status": "HOLD",
            "reason": "multi_fold_stability_failed",
            "stability": stability_result,
            "oos_claimed": False,
            "promotion_authority": "deterministic_research_engine",
        }

    base = holdout.get("baseline", {})
    cand = holdout.get("candidate", {})
    required = ("logloss", "brier", "ece", "accuracy")
    if not isinstance(base, Mapping) or not isinstance(cand, Mapping) or not all(k in base and k in cand for k in required):
        return {"status": "HOLD", "reason": "incomplete_holdout_metrics", "oos_claimed": False}

    try:
        baseline_logloss = float(base["logloss"])
        candidate_logloss = float(cand["logloss"])
        baseline_brier = float(base["brier"])
        candidate_brier = float(cand["brier"])
        baseline_ece = float(base["ece"])
        candidate_ece = float(cand["ece"])
        baseline_accuracy = float(base["accuracy"])
        candidate_accuracy = float(cand["accuracy"])
    except (TypeError, ValueError, OverflowError):
        return {"status": "HOLD", "reason": "non_numeric_holdout_metrics", "oos_claimed": False}

    metrics = (
        baseline_logloss,
        candidate_logloss,
        baseline_brier,
        candidate_brier,
        baseline_ece,
        candidate_ece,
        baseline_accuracy,
        candidate_accuracy,
    )
    if not all(math.isfinite(value) for value in metrics):
        return {"status": "HOLD", "reason": "non_finite_holdout_metrics", "oos_claimed": False}

    primary_relative_improvement = (
        (baseline_logloss - candidate_logloss) / baseline_logloss
        if baseline_logloss > 0
        else None
    )
    auxiliary_relative_improvement = (
        (baseline_brier - candidate_brier) / baseline_brier
        if baseline_brier > 0
        else None
    )
    primary_ok = (
        primary_relative_improvement is not None
        and primary_relative_improvement >= MIN_PRIMARY_RELATIVE_IMPROVEMENT
    )
    auxiliary_ok = (
        auxiliary_relative_improvement is not None
        and auxiliary_relative_improvement >= MIN_AUXILIARY_RELATIVE_IMPROVEMENT
    )
    calibration_delta = candidate_ece - baseline_ece
    calibration_ok = calibration_delta <= MAX_ECE_REGRESSION
    accuracy_not_worse = candidate_accuracy >= baseline_accuracy

    status = "ADOPT" if primary_ok and auxiliary_ok and calibration_ok and accuracy_not_worse else "REJECT"
    return {
        "status": status,
        "primary_logloss_improved": primary_relative_improvement is not None and primary_relative_improvement > 0,
        "auxiliary_brier_improved": auxiliary_relative_improvement is not None and auxiliary_relative_improvement > 0,
        "primary_logloss_relative_improvement": primary_relative_improvement,
        "auxiliary_brier_relative_improvement": auxiliary_relative_improvement,
        "primary_threshold": MIN_PRIMARY_RELATIVE_IMPROVEMENT,
        "auxiliary_threshold": MIN_AUXILIARY_RELATIVE_IMPROVEMENT,
        "calibration_ece_delta": calibration_delta,
        "calibration_ok": calibration_ok,
        "accuracy_not_worse": accuracy_not_worse,
        "development_evidence_present": True,
        "holdout_rows": holdout_rows,
        "holdout_integrity_verified": True,
        "pit_status": pit_status,
        "pit_violations": pit_violations,
        "stability_verified": True,
        "stability": stability_result,
        "promotion_authority": "deterministic_research_engine",
    }


