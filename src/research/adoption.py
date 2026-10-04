from __future__ import annotations

import math

import pandas as pd


MIN_PRIMARY_RELATIVE_IMPROVEMENT = 0.03
MIN_AUXILIARY_RELATIVE_IMPROVEMENT = 0.01
MIN_NON_REGRESSION_FRACTION = 0.70


def _mean_metrics(df: pd.DataFrame) -> pd.Series:
    if "n" in df.columns:
        weight = pd.to_numeric(df["n"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        total = float(weight.sum())
        if total > 0:
            out = {}
            for col in df.columns:
                if col == "n":
                    continue
                values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
                mask = pd.notna(values) & (weight > 0)
                out[col] = float((values[mask] * weight[mask]).sum() / weight[mask].sum()) if mask.any() else float("nan")
            return pd.Series(out)
    return df.mean(numeric_only=True)


def _relative_improvement(baseline: float, candidate: float) -> float | None:
    """Return relative improvement for a lower-is-better metric."""
    base = float(baseline)
    cand = float(candidate)
    if not pd.notna(base) or not pd.notna(cand) or base <= 0:
        return None
    return (base - cand) / base


def _development_non_regression(
    development_oos: pd.DataFrame,
    *,
    max_ece_regression: float,
    max_regression_blocks: int | None,
) -> dict:
    required = {
        "logloss",
        "accuracy",
        "brier",
        "rps",
        "ece",
        "baseline_logistic_logloss",
        "baseline_logistic_accuracy",
        "baseline_logistic_brier",
        "baseline_logistic_rps",
        "baseline_logistic_ece",
    }
    if development_oos.empty:
        return {
            "status": "HOLD",
            "reason": "Missing development OOS blocks",
            "development_blocks": 0,
            "non_regressed_blocks": 0,
            "non_regression_fraction": 0.0,
        }
    if not required.issubset(development_oos.columns):
        return {
            "status": "HOLD",
            "reason": "Development OOS metrics are incomplete",
            "development_blocks": int(len(development_oos)),
            "non_regressed_blocks": 0,
            "non_regression_fraction": 0.0,
            "missing_columns": sorted(required - set(development_oos.columns)),
        }

    non_regressed = 0
    rows = []
    metric_fields = (
        "logloss",
        "accuracy",
        "brier",
        "rps",
        "ece",
        "baseline_logistic_logloss",
        "baseline_logistic_accuracy",
        "baseline_logistic_brier",
        "baseline_logistic_rps",
        "baseline_logistic_ece",
    )
    for block_index, (_, row) in enumerate(development_oos.iterrows()):
        try:
            values = {field: float(row[field]) for field in metric_fields}
        except (TypeError, ValueError, OverflowError):
            return {
                "status": "HOLD",
                "reason": "Development OOS metrics contain non-numeric values",
                "development_blocks": int(len(development_oos)),
                "non_regressed_blocks": 0,
                "non_regression_fraction": 0.0,
                "bad_block": block_index,
            }
        if not all(math.isfinite(value) for value in values.values()):
            return {
                "status": "HOLD",
                "reason": "Development OOS metrics contain non-finite values",
                "development_blocks": int(len(development_oos)),
                "non_regressed_blocks": 0,
                "non_regression_fraction": 0.0,
                "bad_block": block_index,
            }
        checks = {
            "logloss": values["logloss"] <= values["baseline_logistic_logloss"],
            "brier": values["brier"] <= values["baseline_logistic_brier"],
            "rps": values["rps"] <= values["baseline_logistic_rps"],
            "ece": values["ece"] - values["baseline_logistic_ece"] <= max_ece_regression,
            "accuracy": values["accuracy"] >= values["baseline_logistic_accuracy"],
        }
        ok = all(checks.values())
        non_regressed += int(ok)
        rows.append(checks)

    blocks = len(rows)
    fraction = non_regressed / blocks
    allowed_regressions = (
        int(max_regression_blocks)
        if max_regression_blocks is not None
        else int(blocks * (1.0 - MIN_NON_REGRESSION_FRACTION))
    )
    regressions = blocks - non_regressed
    return {
        "status": "PASS" if fraction >= MIN_NON_REGRESSION_FRACTION and regressions <= allowed_regressions else "HOLD",
        "development_blocks": blocks,
        "non_regressed_blocks": non_regressed,
        "regression_blocks": regressions,
        "non_regression_fraction": fraction,
        "required_non_regression_fraction": MIN_NON_REGRESSION_FRACTION,
        "allowed_regression_blocks": allowed_regressions,
        "per_block_checks": rows,
    }


def adoption_decision(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    development_oos: pd.DataFrame | None = None,
    min_accuracy: float = 0.80,
    max_ece_regression: float = 0.02,
    max_regression_blocks: int | None = None,
    min_locked_rows_per_block: int = 500,
) -> dict:
    """Apply the project adoption benchmark to untouched locked chronological OOS.

    The production candidate must beat the same locked OOS baseline by at least
    3% relative LogLoss, improve at least one auxiliary metric by 1% or more,
    avoid regression in calibration/accuracy, and show no metric regression in
    at least 70% of development evaluation blocks.
    """
    required = {"logloss", "accuracy", "brier", "rps", "ece"}
    if baseline.empty or candidate.empty:
        return {"status": "HOLD", "reason": "Missing locked OOS comparison", "oos_claimed": False}
    if not required.issubset(baseline.columns) or not required.issubset(candidate.columns):
        return {"status": "HOLD", "reason": "Locked OOS metrics are incomplete", "oos_claimed": False}
    if "n" not in candidate.columns:
        return {"status": "HOLD", "reason": "Locked OOS block sample sizes are missing", "oos_claimed": False}

    block_sizes = pd.to_numeric(candidate["n"], errors="coerce")
    if not block_sizes.notna().all() or (block_sizes < int(min_locked_rows_per_block)).any():
        return {
            "status": "HOLD",
            "reason": "One or more locked OOS blocks are below the minimum sample size",
            "oos_claimed": False,
            "locked_block_rows": [int(x) if pd.notna(x) else None for x in block_sizes.tolist()],
            "minimum_locked_rows_per_block": int(min_locked_rows_per_block),
        }

    b = _mean_metrics(baseline)
    c = _mean_metrics(candidate)
    n = int(candidate["n"].sum())
    if n <= 0:
        return {"status": "HOLD", "reason": "Locked OOS has no observations", "oos_claimed": False}

    primary_relative_improvement = _relative_improvement(b["logloss"], c["logloss"])
    brier_relative_improvement = _relative_improvement(b["brier"], c["brier"])
    rps_relative_improvement = _relative_improvement(b["rps"], c["rps"])

    primary_ok = (
        primary_relative_improvement is not None
        and primary_relative_improvement >= MIN_PRIMARY_RELATIVE_IMPROVEMENT
    )
    auxiliary_candidates = [x for x in (brier_relative_improvement, rps_relative_improvement) if x is not None]
    auxiliary_best = max(auxiliary_candidates) if auxiliary_candidates else None
    auxiliary_ok = auxiliary_best is not None and auxiliary_best >= MIN_AUXILIARY_RELATIVE_IMPROVEMENT

    ece_delta = float(c["ece"]) - float(b["ece"])
    calibration_ok = ece_delta <= max_ece_regression
    accuracy_not_worse = float(c["accuracy"]) >= float(b["accuracy"])
    development = _development_non_regression(
        development_oos if development_oos is not None else pd.DataFrame(),
        max_ece_regression=max_ece_regression,
        max_regression_blocks=max_regression_blocks,
    )
    status = "ADOPT" if (
        primary_ok
        and auxiliary_ok
        and calibration_ok
        and accuracy_not_worse
        and development["status"] == "PASS"
    ) else "REJECT"

    return {
        "status": status,
        "reason": (
            "Candidate met the project adoption benchmark on untouched locked OOS and development stability."
            if status == "ADOPT"
            else "Candidate did not satisfy the project adoption benchmark."
        ),
        "oos_claimed": True,
        "sample_size": n,
        "baseline": b.to_dict(),
        "candidate": c.to_dict(),
        "delta": {
            "logloss": float(c["logloss"] - b["logloss"]),
            "accuracy": float(c["accuracy"] - b["accuracy"]),
            "brier": float(c["brier"] - b["brier"]),
            "rps": float(c["rps"] - b["rps"]),
            "ece": ece_delta,
        },
        "relative_improvement": {
            "logloss": primary_relative_improvement,
            "brier": brier_relative_improvement,
            "rps": rps_relative_improvement,
            "best_auxiliary": auxiliary_best,
        },
        "checks": {
            "primary_logloss_relative_threshold": MIN_PRIMARY_RELATIVE_IMPROVEMENT,
            "primary_logloss_threshold_met": primary_ok,
            "auxiliary_relative_threshold": MIN_AUXILIARY_RELATIVE_IMPROVEMENT,
            "auxiliary_threshold_met": auxiliary_ok,
            "calibration_ok": calibration_ok,
            "accuracy_not_worse": accuracy_not_worse,
            "accuracy_target": min_accuracy,
            "accuracy_target_met": bool(float(c["accuracy"]) >= min_accuracy),
        },
        "development_stability": development,
        "selection_rule": "validation-only selection; locked OOS untouched",
        "minimum_locked_rows_per_block": int(min_locked_rows_per_block),
        "locked_block_rows": [int(x) for x in block_sizes.tolist()],
    }
