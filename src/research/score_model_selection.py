from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

METHODS = ("primary", "recency", "dixon_coles")
METRICS_LOWER_BETTER = (
    "score_logloss",
    "over_2_5_logloss",
    "over_2_5_brier",
    "btts_logloss",
    "btts_brier",
)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(v[mask], weights=w[mask]))


def _column(method: str, metric: str) -> str:
    return metric if method == "primary" else f"{method}_{metric}"


def _valid_method(frame: pd.DataFrame, method: str) -> bool:
    status_col = None
    if method == "primary":
        status_col = None
    elif method == "recency":
        status_col = "recency_status"
    elif method == "dixon_coles":
        status_col = "dc_status"
    if status_col is not None and status_col in frame.columns:
        if not frame[status_col].astype(str).eq("PASS").all():
            return False
    return all(_column(method, metric) in frame.columns for metric in METRICS_LOWER_BETTER)


def _method_summary(frame: pd.DataFrame, method: str) -> dict[str, Any]:
    weights = pd.to_numeric(frame["n"], errors="coerce") if "n" in frame.columns else pd.Series(1.0, index=frame.index)
    return {
        metric: _weighted_mean(frame[_column(method, metric)], weights)
        for metric in METRICS_LOWER_BETTER
    }


def select_score_model(
    development_oos: pd.DataFrame,
    *,
    min_blocks: int = 2,
    min_relative_improvement: float = 0.005,
    max_metric_regression: float = 0.01,
    min_improvement_share: float = 2.0 / 3.0,
) -> dict[str, Any]:
    """Choose a score challenger using development OOS only.

    Primary is always available. A challenger must:
    1) have complete finite development metrics,
    2) beat primary score LogLoss by a meaningful relative margin,
    3) not regress any tracked secondary proper score beyond tolerance,
    4) improve on at least the configured share of development blocks.

    Locked OOS is never inspected here.
    """
    if development_oos.empty:
        return {
            "selected_method": "primary",
            "status": "HOLD",
            "reason": "No development OOS rows available",
            "evaluated_blocks": 0,
        }
    if "n" not in development_oos.columns:
        work = development_oos.copy()
        work["n"] = 1.0
    else:
        work = development_oos.copy()

    primary = _method_summary(work, "primary")
    if not np.isfinite(primary["score_logloss"]):
        return {
            "selected_method": "primary",
            "status": "HOLD",
            "reason": "Primary score development LogLoss is non-finite",
            "evaluated_blocks": int(len(work)),
        }

    best_method = "primary"
    best_reason = "No challenger satisfied development OOS gate"
    candidate_records: dict[str, dict[str, Any]] = {
        "primary": {"status": "PRIMARY", "metrics": primary}
    }

    required_blocks = max(int(min_blocks), 1)
    for method in METHODS[1:]:
        if len(work) < required_blocks or not _valid_method(work, method):
            candidate_records[method] = {
                "status": "UNAVAILABLE",
                "metrics": {},
            }
            continue

        summary = _method_summary(work, method)
        if not all(np.isfinite(summary[m]) for m in METRICS_LOWER_BETTER):
            candidate_records[method] = {
                "status": "REJECT",
                "reason": "Non-finite development metric",
                "metrics": summary,
            }
            continue

        p_loss = primary["score_logloss"]
        c_loss = summary["score_logloss"]
        relative_gain = float((p_loss - c_loss) / max(abs(p_loss), 1e-9))

        block_improved = []
        for _, row in work.iterrows():
            base = pd.to_numeric(row[_column("primary", "score_logloss")], errors="coerce")
            challenger = pd.to_numeric(row[_column(method, "score_logloss")], errors="coerce")
            block_improved.append(bool(np.isfinite(base) and np.isfinite(challenger) and challenger < base))
        improvement_share = float(np.mean(block_improved)) if block_improved else 0.0

        regressions = {}
        for metric in METRICS_LOWER_BETTER:
            base = primary[metric]
            cand = summary[metric]
            regressions[metric] = float((cand - base) / max(abs(base), 1e-9))

        secondary_ok = all(v <= float(max_metric_regression) for k, v in regressions.items() if k != "score_logloss")
        score_ok = relative_gain >= float(min_relative_improvement)
        share_ok = improvement_share >= float(min_improvement_share)
        accepted = bool(score_ok and secondary_ok and share_ok)

        candidate_records[method] = {
            "status": "ACCEPT" if accepted else "REJECT",
            "metrics": summary,
            "relative_score_logloss_gain": relative_gain,
            "improvement_share": improvement_share,
            "secondary_relative_regressions": regressions,
            "checks": {
                "score_logloss_gain_ok": score_ok,
                "secondary_regression_ok": secondary_ok,
                "block_improvement_share_ok": share_ok,
            },
        }

        if accepted:
            if best_method == "primary" or c_loss < _method_summary(work, best_method)["score_logloss"]:
                best_method = method
                best_reason = (
                    f"{method} improved development score LogLoss with stable block-level gains "
                    "and no material tracked secondary regression"
                )

    return {
        "selected_method": best_method,
        "status": "ADOPT_CANDIDATE" if best_method != "primary" else "KEEP_PRIMARY",
        "reason": best_reason,
        "evaluated_blocks": int(len(work)),
        "candidate_records": candidate_records,
        "selection_rule": {
            "development_oos_only": True,
            "locked_oos_inspected": False,
            "min_relative_improvement": float(min_relative_improvement),
            "max_metric_regression": float(max_metric_regression),
            "min_improvement_share": float(min_improvement_share),
        },
    }
