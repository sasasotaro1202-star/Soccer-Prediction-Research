from __future__ import annotations

import pandas as pd


def _mean_metrics(df: pd.DataFrame) -> pd.Series:
    return df.mean(numeric_only=True)


def adoption_decision(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    development_oos: pd.DataFrame | None = None,
    min_accuracy: float = 0.80,
    max_ece_regression: float = 0.02,
    max_regression_blocks: int = 0,
) -> dict:
    """Strict adoption gate for a candidate evaluated on an untouched locked OOS.

    The locked block is never used for feature/model selection. Development OOS is
    used only as a stability diagnostic; it cannot replace the locked comparison.
    Accuracy >= 80% is a target/reporting criterion, not a reason to tune on OOS.
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
        # Each development block was itself generated strictly after its training
        # window. We use it only to detect broad regressions, never to tune weights.
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
