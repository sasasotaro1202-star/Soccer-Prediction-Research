from __future__ import annotations

import pandas as pd


def _mean_metrics(df: pd.DataFrame) -> pd.Series:
    """Aggregate block metrics by match count, not by block count.

    Walk-forward blocks can have different sample sizes. Equal-weighting blocks
    can therefore let a small tail block dominate a large historical block.
    Match-count weighting is the correct aggregation for the locked OOS gate.
    """
    numeric = df.select_dtypes(include="number")
    if "n" in numeric.columns and numeric["n"].sum() > 0:
        weights = numeric["n"].astype(float)
        total = float(weights.sum())
        out = {}
        for col in numeric.columns:
            if col == "n":
                out[col] = total
                continue
            values = pd.to_numeric(numeric[col], errors="coerce")
            mask = values.notna() & weights.notna()
            out[col] = float((values[mask] * weights[mask]).sum() / weights[mask].sum()) if mask.any() else float("nan")
        return pd.Series(out)
    return numeric.mean()


def adoption_decision(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    development_oos: pd.DataFrame | None = None,
    min_accuracy: float = 0.80,
    max_ece_regression: float = 0.02,
    max_regression_blocks: int = 0,
) -> dict:
    """Strict adoption gate for a candidate evaluated on untouched locked OOS.

    Locked OOS is never used for feature/model selection. Development OOS is
    only a stability diagnostic. Accuracy >= 80% is a target/reporting metric,
    not a tuning knob, so a candidate cannot be forced to pass by overfitting.
    """
    required = {"logloss", "accuracy", "brier", "ece"}
    if baseline.empty or candidate.empty:
        return {"status": "HOLD", "reason": "Missing locked OOS comparison", "oos_claimed": False}
    if not required.issubset(baseline.columns) or not required.issubset(candidate.columns):
        return {"status": "HOLD", "reason": "Locked OOS metrics are incomplete", "oos_claimed": False}

    b = _mean_metrics(baseline)
    c = _mean_metrics(candidate)
    n = int(candidate["n"].sum()) if "n" in candidate.columns else 0
    if n <= 0:
        return {"status": "HOLD", "reason": "Locked OOS has no observations", "oos_claimed": False}

    logloss_improved = float(c["logloss"]) < float(b["logloss"])
    brier_not_worse = float(c["brier"]) <= float(b["brier"])
    ece_delta = float(c["ece"]) - float(b["ece"])
    calibration_ok = ece_delta <= max_ece_regression
    accuracy_not_worse = float(c["accuracy"]) >= float(b["accuracy"])

    stability = {
        "status": "NOT_AVAILABLE",
        "development_blocks": 0,
        "regression_blocks": 0,
        "required_max_regression_blocks": max_regression_blocks,
    }
    if development_oos is not None and not development_oos.empty:
        required_dev = {
            "logloss", "accuracy", "brier", "ece",
            "baseline_logistic_logloss", "baseline_logistic_accuracy",
            "baseline_logistic_brier", "baseline_logistic_ece",
        }
        if required_dev.issubset(development_oos.columns):
            regression_blocks = 0
            for _, row in development_oos.iterrows():
                if not (
                    float(row["logloss"]) < float(row["baseline_logistic_logloss"])
                    and float(row["brier"]) <= float(row["baseline_logistic_brier"])
                    and float(row["ece"]) - float(row["baseline_logistic_ece"]) <= max_ece_regression
                    and float(row["accuracy"]) >= float(row["baseline_logistic_accuracy"])
                ):
                    regression_blocks += 1
            stability = {
                "status": "PASS" if regression_blocks <= max_regression_blocks else "FAIL",
                "development_blocks": int(len(development_oos)),
                "regression_blocks": int(regression_blocks),
                "required_max_regression_blocks": int(max_regression_blocks),
            }

    status = "ADOPT" if (
        logloss_improved
        and brier_not_worse
        and calibration_ok
        and accuracy_not_worse
        and stability["status"] in {"PASS", "NOT_AVAILABLE"}
    ) else "REJECT"

    return {
        "status": status,
        "reason": (
            "Candidate improved the untouched locked OOS and passed development stability checks."
            if status == "ADOPT"
            else "Candidate did not satisfy the locked OOS and stability adoption gate."
        ),
        "oos_claimed": True,
        "sample_size": n,
        "aggregation": "match_count_weighted_across_oos_blocks",
        "baseline": b.to_dict(),
        "candidate": c.to_dict(),
        "delta": {
            "logloss": float(c["logloss"] - b["logloss"]),
            "accuracy": float(c["accuracy"] - b["accuracy"]),
            "brier": float(c["brier"] - b["brier"]),
            "ece": ece_delta,
        },
        "checks": {
            "logloss_improved": logloss_improved,
            "brier_not_worse": brier_not_worse,
            "calibration_ok": calibration_ok,
            "accuracy_not_worse": accuracy_not_worse,
            "accuracy_target": min_accuracy,
            "accuracy_target_met": bool(float(c["accuracy"]) >= min_accuracy),
        },
        "stability": stability,
        "selection_rule": "validation-only selection; locked OOS untouched",
    }
