"""Research-only robustness diagnostic for statsmodels Score OOS."""
from __future__ import annotations

import math
from typing import Any, Mapping


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def evaluate_statsmodels_robustness(
    oos_result: Mapping[str, Any],
    *,
    min_folds: int = 3,
    min_competition_rows: int = 100,
    min_non_regression_fraction: float = 0.70,
    max_material_regression_fraction: float = 0.05,
) -> dict[str, Any]:
    """Evaluate robustness without touching selection or Frozen Holdout.

    The diagnostic is deliberately conservative:
    - every fold must have valid PIT/training/calibration flags;
    - calibrated candidate-vs-incumbent Score LogLoss must not regress in more
      than 30% of eligible folds;
    - eligible competition slices must not show >5% material deterioration in
      more than 30% of slices;
    - unsupported/OOD coverage is reported separately, never treated as zero.
    """
    if not isinstance(oos_result, Mapping):
        return {"status": "HOLD", "reason": "invalid_oos_result"}

    if oos_result.get("status") != "RESEARCH_OOS_READY":
        return {
            "status": "HOLD",
            "reason": "oos_result_not_ready",
            "oos_status": oos_result.get("status"),
        }

    rows = oos_result.get("rows")
    if not isinstance(rows, list) or len(rows) < int(min_folds):
        return {
            "status": "HOLD",
            "reason": "too_few_oos_folds",
            "fold_count": len(rows) if isinstance(rows, list) else 0,
            "required_min_folds": int(min_folds),
        }

    fold_deltas = []
    coverage_values = []
    fold_failures = []
    eligible_competitions = []
    material_comp_regressions = []

    for i, row in enumerate(rows):
        required_flags = (
            "pit_training_boundary_valid",
            "pit_oos_case_valid",
            "same_kickoff_split_avoided",
            "calibration_precedes_oos",
        )
        if not all(row.get(flag) is True for flag in required_flags):
            fold_failures.append(i)
            continue

        base = row.get("incumbent_calibrated_score_logloss")
        cand = row.get("statsmodels_calibrated_score_logloss")
        if not _finite(base) or not _finite(cand):
            fold_failures.append(i)
            continue

        fold_deltas.append(float(cand) - float(base))
        coverage = row.get("common_coverage")
        if _finite(coverage):
            coverage_values.append(float(coverage))

        for slice_row in row.get("competition_slices", []):
            try:
                n = int(slice_row.get("n", 0))
            except (TypeError, ValueError):
                continue
            if n < int(min_competition_rows):
                continue
            base_slice = slice_row.get("incumbent_calibrated_score_logloss")
            cand_slice = slice_row.get("statsmodels_calibrated_score_logloss")
            if not _finite(base_slice) or not _finite(cand_slice) or float(base_slice) <= 0:
                continue
            relative_regression = (float(cand_slice) - float(base_slice)) / float(base_slice)
            eligible_competitions.append(
                {
                    "competition": str(slice_row.get("competition", "")),
                    "n": n,
                    "relative_regression": relative_regression,
                }
            )
            material_comp_regressions.append(relative_regression > float(max_material_regression_fraction))

    if fold_failures:
        return {
            "status": "HOLD",
            "reason": "invalid_fold_integrity",
            "invalid_folds": fold_failures,
            "fold_count": len(rows),
            "selection_performed": False,
            "frozen_holdout_used": False,
        }

    fold_non_regressed = sum(delta <= 0 for delta in fold_deltas)
    fold_fraction = fold_non_regressed / max(1, len(fold_deltas))
    comp_material = sum(1 for flag in material_comp_regressions if flag)
    comp_total = len(material_comp_regressions)
    comp_fraction = (
        comp_material / comp_total
        if comp_total
        else 0.0
    )

    status = (
        "PASS"
        if (
            fold_fraction >= float(min_non_regression_fraction)
            and comp_fraction <= 1.0 - float(min_non_regression_fraction)
        )
        else "HOLD"
    )

    return {
        "status": status,
        "reason": (
            "chronological fold and competition robustness thresholds passed"
            if status == "PASS"
            else "robustness thresholds not satisfied"
        ),
        "fold_count": int(len(rows)),
        "folds_non_regressed": int(fold_non_regressed),
        "fold_non_regression_fraction": float(fold_fraction),
        "required_non_regression_fraction": float(min_non_regression_fraction),
        "competition_slice_count": int(comp_total),
        "competition_material_regression_count": int(comp_material),
        "competition_material_regression_fraction": float(comp_fraction),
        "max_material_regression_fraction": float(max_material_regression_fraction),
        "min_competition_rows": int(min_competition_rows),
        "coverage_min": float(min(coverage_values)) if coverage_values else None,
        "coverage_mean": float(sum(coverage_values) / len(coverage_values)) if coverage_values else None,
        "coverage_max": float(max(coverage_values)) if coverage_values else None,
        "unsupported_rows_total": int(sum(int(row.get("unsupported_rows", 0)) for row in rows)),
        "selection_performed": False,
        "frozen_holdout_used": False,
        "production_usable": False,
    }
