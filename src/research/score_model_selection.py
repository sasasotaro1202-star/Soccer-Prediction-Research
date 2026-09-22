from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

METHODS = ("primary", "recency", "dixon_coles")
METRICS = ("score_logloss", "over_2_5_logloss", "over_2_5_brier", "btts_logloss", "btts_brier")

def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float(np.average(v[mask], weights=w[mask])) if mask.any() else float("nan")

def _col(method: str, metric: str) -> str:
    return metric if method == "primary" else f"{method}_{metric}"

def _summary(frame: pd.DataFrame, method: str) -> dict[str, float]:
    weights = pd.to_numeric(frame.get("n", pd.Series(1.0, index=frame.index)), errors="coerce").fillna(0.0)
    return {metric: _weighted_mean(frame[_col(method, metric)], weights) for metric in METRICS}

def select_score_model(
    development_oos: pd.DataFrame,
    *,
    min_blocks: int = 2,
    min_relative_improvement: float = 0.005,
    max_metric_regression: float = 0.01,
    min_improvement_share: float = 2.0 / 3.0,
) -> dict[str, Any]:
    """Select a score method from development OOS only; locked OOS is never inspected."""
    if development_oos.empty:
        return {"selected_method": "primary", "status": "HOLD", "reason": "No development OOS rows available", "evaluated_blocks": 0}
    work = development_oos.copy()
    primary = _summary(work, "primary")
    if not np.isfinite(primary["score_logloss"]):
        return {"selected_method": "primary", "status": "HOLD", "reason": "Primary score LogLoss is non-finite", "evaluated_blocks": int(len(work))}

    records: dict[str, dict[str, Any]] = {"primary": {"status": "PRIMARY", "metrics": primary}}
    selected = "primary"
    selected_loss = primary["score_logloss"]

    for method in METHODS[1:]:
        status_col = "recency_status" if method == "recency" else "dc_status"
        if status_col not in work.columns or not work[status_col].astype(str).eq("PASS").all():
            records[method] = {"status": "UNAVAILABLE", "metrics": {}}
            continue
        required = {_col(method, metric) for metric in METRICS}
        if len(work) < max(min_blocks, 1) or not required.issubset(work.columns):
            records[method] = {"status": "UNAVAILABLE", "metrics": {}}
            continue
        summary = _summary(work, method)
        if not all(np.isfinite(summary[m]) for m in METRICS):
            records[method] = {"status": "REJECT", "reason": "Non-finite development metric", "metrics": summary}
            continue

        gains = []
        for _, row in work.iterrows():
            base = pd.to_numeric(row[_col("primary", "score_logloss")], errors="coerce")
            cand = pd.to_numeric(row[_col(method, "score_logloss")], errors="coerce")
            gains.append(bool(np.isfinite(base) and np.isfinite(cand) and cand < base))
        share = float(np.mean(gains)) if gains else 0.0
        relative_gain = float((primary["score_logloss"] - summary["score_logloss"]) / max(abs(primary["score_logloss"]), 1e-9))
        regressions = {
            m: float((summary[m] - primary[m]) / max(abs(primary[m]), 1e-9))
            for m in METRICS
        }
        checks = {
            "score_logloss_gain_ok": relative_gain >= min_relative_improvement,
            "secondary_regression_ok": all(regressions[m] <= max_metric_regression for m in METRICS if m != "score_logloss"),
            "block_improvement_share_ok": share >= min_improvement_share,
        }
        accepted = all(checks.values())
        records[method] = {
            "status": "ACCEPT" if accepted else "REJECT",
            "metrics": summary,
            "relative_score_logloss_gain": relative_gain,
            "improvement_share": share,
            "secondary_relative_regressions": regressions,
            "checks": checks,
        }
        if accepted and summary["score_logloss"] < selected_loss:
            selected = method
            selected_loss = summary["score_logloss"]

    return {
        "selected_method": selected,
        "status": "ADOPT_CANDIDATE" if selected != "primary" else "KEEP_PRIMARY",
        "reason": "Development-OOS challenger gate passed" if selected != "primary" else "No score challenger passed the development-OOS gate",
        "evaluated_blocks": int(len(work)),
        "candidate_records": records,
        "selection_rule": {
            "development_oos_only": True,
            "locked_oos_inspected": False,
            "min_relative_improvement": min_relative_improvement,
            "max_metric_regression": max_metric_regression,
            "min_improvement_share": min_improvement_share,
        },
    }
