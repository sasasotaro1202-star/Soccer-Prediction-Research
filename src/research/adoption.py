from __future__ import annotations

import pandas as pd


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


def adoption_decision(baseline: pd.DataFrame, candidate: pd.DataFrame, *, development_oos: pd.DataFrame | None = None, min_accuracy: float = 0.80, max_ece_regression: float = 0.02, max_regression_blocks: int = 0, min_locked_rows_per_block: int = 500) -> dict:
    """Strict adoption gate using untouched locked OOS plus development stability."""
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
    b = _mean_metrics(baseline); c = _mean_metrics(candidate)
    n = int(candidate["n"].sum()) if "n" in candidate.columns else 0
    if n <= 0:
        return {"status": "HOLD", "reason": "Locked OOS has no observations", "oos_claimed": False}
    logloss_improved = float(c["logloss"]) < float(b["logloss"])
    brier_not_worse = float(c["brier"]) <= float(b["brier"])
    rps_not_worse = float(c["rps"]) <= float(b["rps"])
    ece_delta = float(c["ece"]) - float(b["ece"])
    calibration_ok = ece_delta <= max_ece_regression
    accuracy_not_worse = float(c["accuracy"]) >= float(b["accuracy"])
    stability = {"status": "NOT_AVAILABLE", "development_blocks": 0, "regression_blocks": 0, "required_max_regression_blocks": max_regression_blocks}
    if development_oos is not None and not development_oos.empty:
        required_dev = {"logloss", "accuracy", "brier", "rps", "ece", "baseline_logistic_logloss", "baseline_logistic_accuracy", "baseline_logistic_brier", "baseline_logistic_rps", "baseline_logistic_ece"}
        if required_dev.issubset(development_oos.columns):
            regression_blocks = 0
            for _, row in development_oos.iterrows():
                if not (float(row["logloss"]) < float(row["baseline_logistic_logloss"]) and float(row["brier"]) <= float(row["baseline_logistic_brier"]) and float(row["rps"]) <= float(row["baseline_logistic_rps"]) and float(row["ece"]) - float(row["baseline_logistic_ece"]) <= max_ece_regression and float(row["accuracy"]) >= float(row["baseline_logistic_accuracy"])): regression_blocks += 1
            stability = {"status": "PASS" if regression_blocks <= max_regression_blocks else "FAIL", "development_blocks": int(len(development_oos)), "regression_blocks": int(regression_blocks), "required_max_regression_blocks": int(max_regression_blocks)}
    status = "ADOPT" if (logloss_improved and brier_not_worse and rps_not_worse and calibration_ok and accuracy_not_worse and stability["status"] == "PASS") else "REJECT"
    return {"status": status, "reason": "Candidate improved the untouched locked OOS and passed development stability checks." if status == "ADOPT" else "Candidate did not satisfy the locked OOS and stability adoption gate.", "oos_claimed": True, "sample_size": n, "baseline": b.to_dict(), "candidate": c.to_dict(), "delta": {"logloss": float(c["logloss"] - b["logloss"]), "accuracy": float(c["accuracy"] - b["accuracy"]), "brier": float(c["brier"] - b["brier"]), "rps": float(c["rps"] - b["rps"]), "ece": ece_delta}, "checks": {"logloss_improved": logloss_improved, "brier_not_worse": brier_not_worse, "rps_not_worse": rps_not_worse, "calibration_ok": calibration_ok, "accuracy_not_worse": accuracy_not_worse, "accuracy_target": min_accuracy, "accuracy_target_met": bool(float(c["accuracy"]) >= min_accuracy)}, "stability": stability, "selection_rule": "validation-only selection; locked OOS untouched", "minimum_locked_rows_per_block": int(min_locked_rows_per_block), "locked_block_rows": [int(x) for x in block_sizes.tolist()]}
