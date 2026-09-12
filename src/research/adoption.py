from __future__ import annotations

import pandas as pd


def adoption_decision(baseline: pd.DataFrame, candidate: pd.DataFrame, *, min_accuracy: float = 0.80, max_ece_regression: float = 0.02) -> dict:
    """Strict PIT-safe adoption gate for an untouched OOS comparison.

    Model/feature choices must be made from historical validation only. This
    function is called after the candidate is locked and therefore cannot be
    used to tune the locked OOS block.
    """
    if baseline.empty or candidate.empty:
        return {"status": "HOLD", "reason": "Missing locked OOS comparison", "oos_claimed": False}
    required = {"logloss", "accuracy", "brier", "ece"}
    if not required.issubset(baseline.columns) or not required.issubset(candidate.columns):
        return {"status": "HOLD", "reason": "Locked OOS metrics are incomplete", "oos_claimed": False}
    b = baseline.mean(numeric_only=True); c = candidate.mean(numeric_only=True)
    n = int(candidate["n"].sum()) if "n" in candidate.columns else 0
    if n <= 0:
        return {"status": "HOLD", "reason": "Locked OOS has no observations", "oos_claimed": False}
    logloss_improved = float(c["logloss"]) < float(b["logloss"])
    brier_not_worse = float(c["brier"]) <= float(b["brier"])
    ece_delta = float(c["ece"]) - float(b["ece"])
    calibration_ok = ece_delta <= max_ece_regression
    accuracy_not_worse = float(c["accuracy"]) >= float(b["accuracy"])
    status = "ADOPT" if logloss_improved and brier_not_worse and calibration_ok and accuracy_not_worse else "REJECT"
    return {
        "status": status,
        "reason": "Candidate improved locked OOS without unacceptable calibration/Brier regression." if status == "ADOPT" else "Candidate did not satisfy the locked OOS adoption gate.",
        "oos_claimed": True,
        "sample_size": n,
        "baseline": b.to_dict(),
        "candidate": c.to_dict(),
        "delta": {"logloss": float(c["logloss"]-b["logloss"]), "accuracy": float(c["accuracy"]-b["accuracy"]), "brier": float(c["brier"]-b["brier"]), "ece": ece_delta},
        "checks": {"logloss_improved": logloss_improved, "brier_not_worse": brier_not_worse, "calibration_ok": calibration_ok, "accuracy_not_worse": accuracy_not_worse, "accuracy_target": min_accuracy, "accuracy_target_met": bool(float(c["accuracy"]) >= min_accuracy)},
        "selection_rule": "validation-only selection; locked OOS untouched",
    }
