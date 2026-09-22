from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

METHODS = ("primary", "recency", "time_decay", "dixon_coles", "negative_binomial")
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
    min_blocks: int = 3,
    min_relative_improvement: float = 0.005,
    max_metric_regression: float = 0.01,
    min_improvement_share: float = 2.0 / 3.0,
    min_rows_per_block: int = 500,
) -> dict[str, Any]:
    """Select a score method from development OOS only; locked OOS is never inspected."""
    if development_oos.empty:
        return {"selected_method": "primary", "status": "HOLD", "reason": "No development OOS rows available", "evaluated_blocks": 0}
    work = development_oos.copy()
    if "n" not in work.columns:
        return {
            "selected_method": "primary",
            "status": "HOLD",
            "reason": "Development OOS block sample sizes are missing",
            "evaluated_blocks": int(len(work)),
        }
    block_sizes = pd.to_numeric(work["n"], errors="coerce")
    if not block_sizes.notna().all() or (block_sizes < int(min_rows_per_block)).any():
        return {
            "selected_method": "primary",
            "status": "HOLD",
            "reason": "One or more development OOS blocks are below the minimum sample size",
            "evaluated_blocks": int(len(work)),
            "development_block_rows": [int(x) if pd.notna(x) else None for x in block_sizes.tolist()],
            "minimum_rows_per_block": int(min_rows_per_block),
        }
    primary = _summary(work, "primary")
    if not np.isfinite(primary["score_logloss"]):
        return {"selected_method": "primary", "status": "HOLD", "reason": "Primary score LogLoss is non-finite", "evaluated_blocks": int(len(work))}

    records: dict[str, dict[str, Any]] = {"primary": {"status": "PRIMARY", "metrics": primary}}
    selected = "primary"
    selected_loss = primary["score_logloss"]

    for method in METHODS[1:]:
        status_col = {
            "recency": "recency_status",
            "time_decay": "time_decay_status",
            "dixon_coles": "dc_status",
            "negative_binomial": "negative_binomial_status",
        }[method]
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
            "min_rows_per_block": int(min_rows_per_block),
            "development_block_rows": [int(x) for x in block_sizes.tolist()],
        },
    }



def verify_selected_score_model(
    selection: dict[str, Any],
    locked_oos: pd.DataFrame,
    *,
    max_metric_regression: float = 0.02,
    min_rows_per_block: int = 500,
) -> dict[str, Any]:
    """Evaluate the development-selected score method on untouched locked OOS."""

    selected = str(selection.get("selected_method", "primary"))
    if locked_oos.empty:
        return {
            "selected_method": selected,
            "status": "HOLD",
            "reason": "Locked OOS is empty",
            "locked_oos_inspected": True,
        }

    if "n" not in locked_oos.columns:
        return {
            "selected_method": "primary",
            "status": "HOLD",
            "reason": "Locked OOS block sample sizes are missing",
            "locked_oos_inspected": True,
        }
    locked_block_sizes = pd.to_numeric(locked_oos["n"], errors="coerce")
    if not locked_block_sizes.notna().all() or (locked_block_sizes < int(min_rows_per_block)).any():
        return {
            "selected_method": "primary" if selected != "primary" else selected,
            "status": "HOLD",
            "reason": "One or more locked OOS blocks are below the minimum sample size",
            "locked_oos_inspected": True,
            "locked_block_rows": [int(x) if pd.notna(x) else None for x in locked_block_sizes.tolist()],
            "minimum_rows_per_block": int(min_rows_per_block),
        }

    required_all = {"n"} | {_col("primary", metric) for metric in METRICS}
    if not required_all.issubset(locked_oos.columns):
        return {
            "selected_method": "primary" if selected != "primary" else selected,
            "status": "REJECT",
            "reason": "Locked OOS score metrics are incomplete",
            "locked_oos_inspected": True,
            "missing_columns": sorted(required_all - set(locked_oos.columns)),
        }

    base = _summary(locked_oos, "primary")
    if not all(np.isfinite(base[m]) for m in METRICS):
        return {
            "selected_method": "primary",
            "status": "REJECT",
            "reason": "Primary locked OOS metric is non-finite",
            "locked_oos_inspected": True,
            "baseline_metrics": base,
        }

    if selected == "primary":
        return {
            "selected_method": "primary",
            "status": "PASS",
            "reason": "Primary retained with finite untouched locked OOS evidence for score/O-U/BTTS metrics",
            "locked_oos_inspected": True,
            "locked_oos_blocks": int(len(locked_oos)),
            "locked_block_rows": [int(x) for x in locked_block_sizes.tolist()],
            "minimum_rows_per_block": int(min_rows_per_block),
            "baseline_metrics": base,
            "selected_metrics": base,
            "relative_deltas": {metric: 0.0 for metric in METRICS},
            "checks": {
                "all_required_locked_metrics_finite": True,
                "primary_self_consistent": True,
            },
        }

    if selected not in METHODS[1:]:
        return {
            "selected_method": "primary",
            "status": "REJECT",
            "reason": f"Unknown selected score method: {selected}",
            "locked_oos_inspected": True,
            "baseline_metrics": base,
        }

    required = {"n"} | {_col(selected, metric) for metric in METRICS}
    if not required.issubset(locked_oos.columns):
        return {
            "selected_method": "primary",
            "status": "REJECT",
            "reason": "Selected challenger locked OOS metrics are incomplete",
            "locked_oos_inspected": True,
            "missing_columns": sorted(required - set(locked_oos.columns)),
        }

    status_col = {
        "recency": "recency_status",
        "time_decay": "time_decay_status",
        "dixon_coles": "dc_status",
        "negative_binomial": "negative_binomial_status",
    }.get(selected)
    if status_col in locked_oos.columns and not locked_oos[status_col].astype(str).eq("PASS").all():
        return {
            "selected_method": "primary",
            "status": "REJECT",
            "reason": f"{selected} is unavailable on locked OOS",
            "locked_oos_inspected": True,
            "baseline_metrics": base,
        }

    cand = _summary(locked_oos, selected)
    if not all(np.isfinite(base[m]) and np.isfinite(cand[m]) for m in METRICS):
        return {
            "selected_method": "primary",
            "status": "REJECT",
            "reason": "Non-finite locked OOS metric",
            "locked_oos_inspected": True,
            "baseline_metrics": base,
            "selected_metrics": cand,
        }

    relative_deltas = {
        m: float((cand[m] - base[m]) / max(abs(base[m]), 1e-9))
        for m in METRICS
    }
    block_ok = []
    for _, row in locked_oos.iterrows():
        bp = pd.to_numeric(row["score_logloss"], errors="coerce")
        cp = pd.to_numeric(row[_col(selected, "score_logloss")], errors="coerce")
        block_ok.append(
            bool(
                np.isfinite(bp)
                and np.isfinite(cp)
                and cp <= bp * (1.0 + float(max_metric_regression))
            )
        )
    overall_score_ok = relative_deltas["score_logloss"] <= 0.0
    secondary_ok = all(
        relative_deltas[m] <= float(max_metric_regression)
        for m in METRICS
        if m != "score_logloss"
    )
    blocks_ok = bool(block_ok) and all(block_ok)
    passed = bool(overall_score_ok and secondary_ok and blocks_ok)

    return {
        "selected_method": selected if passed else "primary",
        "status": "PASS" if passed else "REJECT",
        "reason": "Selected challenger survived untouched locked OOS" if passed else "Selected challenger failed locked OOS verification",
        "locked_oos_inspected": True,
        "locked_oos_blocks": int(len(locked_oos)),
        "locked_block_rows": [int(x) for x in locked_block_sizes.tolist()],
        "minimum_rows_per_block": int(min_rows_per_block),
        "baseline_metrics": base,
        "selected_metrics": cand,
        "relative_deltas": relative_deltas,
        "checks": {
            "all_required_locked_metrics_finite": True,
            "overall_score_logloss_not_worse": overall_score_ok,
            "secondary_metrics_not_materially_worse": secondary_ok,
            "block_level_score_logloss_not_materially_worse": blocks_ok,
        },
    }
