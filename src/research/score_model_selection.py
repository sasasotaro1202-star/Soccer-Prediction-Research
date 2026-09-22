from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


METHODS = ("primary", "recency", "dixon_coles")
RECENCY_HALF_LIVES_DAYS = (180.0, 365.0, 730.0, 1095.0)
METRICS = (
    "score_logloss",
    "over_2_5_logloss",
    "over_2_5_brier",
    "btts_logloss",
    "btts_brier",
)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    positive = w > 0
    if not positive.any():
        return float("nan")
    if not np.isfinite(v[positive]).all() or not np.isfinite(w[positive]).all():
        return float("nan")
    return float(np.average(v[positive], weights=w[positive]))


def _col(
    method: str,
    metric: str,
    *,
    half_life_days: float | None = None,
    half_life_rows: float | None = None,
) -> str:
    if method == "primary":
        return metric
    if method == "recency" and half_life_days is not None:
        return f"recency_d{int(half_life_days)}_{metric}"
    if method == "recency" and half_life_rows is not None:
        return f"recency_h{int(half_life_rows)}_{metric}"
    return f"{method}_{metric}"


def _summary(
    frame: pd.DataFrame,
    method: str,
    *,
    half_life_days: float | None = None,
    half_life_rows: float | None = None,
) -> dict[str, float]:
    weights = pd.to_numeric(
        frame.get("n", pd.Series(1.0, index=frame.index)), errors="coerce"
    ).fillna(0.0)
    cols = {
        _col(
            method,
            metric,
            half_life_days=half_life_days,
            half_life_rows=half_life_rows,
        )
        for metric in METRICS
    }
    if not cols.issubset(frame.columns):
        raise KeyError(f"Missing score metrics: {sorted(cols - set(frame.columns))}")
    return {
        metric: _weighted_mean(
            frame[
                _col(
                    method,
                    metric,
                    half_life_days=half_life_days,
                    half_life_rows=half_life_rows,
                )
            ],
            weights,
        )
        for metric in METRICS
    }


def _recency_specs(frame: pd.DataFrame) -> list[tuple[str, float]]:
    return [
        (f"recency_d{int(half)}", half)
        for half in RECENCY_HALF_LIVES_DAYS
        if f"recency_d{int(half)}_score_logloss" in frame.columns
    ]


def _check_challenger(
    work: pd.DataFrame,
    primary: dict[str, float],
    method: str,
    *,
    half_life_days: float | None = None,
    half_life_rows: float | None = None,
    max_metric_regression: float,
    min_relative_improvement: float,
    min_improvement_share: float,
) -> tuple[dict[str, Any], bool]:
    summary = _summary(
        work,
        method,
        half_life_days=half_life_days,
        half_life_rows=half_life_rows,
    )
    if not all(np.isfinite(summary[m]) for m in METRICS):
        return {
            "status": "REJECT",
            "reason": "Non-finite development metric",
            "metrics": summary,
        }, False

    gains = []
    for _, row in work.iterrows():
        base = pd.to_numeric(row["score_logloss"], errors="coerce")
        cand = pd.to_numeric(
            row[
                _col(
                    method,
                    "score_logloss",
                    half_life_days=half_life_days,
                    half_life_rows=half_life_rows,
                )
            ],
            errors="coerce",
        )
        gains.append(bool(np.isfinite(base) and np.isfinite(cand) and cand < base))

    share = float(np.mean(gains)) if gains else 0.0
    relative_gain = float(
        (primary["score_logloss"] - summary["score_logloss"])
        / max(abs(primary["score_logloss"]), 1e-9)
    )
    regressions = {
        m: float((summary[m] - primary[m]) / max(abs(primary[m]), 1e-9))
        for m in METRICS
    }
    checks = {
        "score_logloss_gain_ok": relative_gain >= min_relative_improvement,
        "secondary_regression_ok": all(
            regressions[m] <= max_metric_regression
            for m in METRICS
            if m != "score_logloss"
        ),
        "block_improvement_share_ok": share >= min_improvement_share,
    }
    accepted = all(checks.values())
    return {
        "status": "ACCEPT" if accepted else "REJECT",
        "metrics": summary,
        "relative_score_logloss_gain": relative_gain,
        "improvement_share": share,
        "secondary_relative_regressions": regressions,
        "checks": checks,
        "parameters": (
            {"half_life_days": float(half_life_days)}
            if method == "recency" and half_life_days is not None
            else (
                {"half_life_rows": float(half_life_rows)}
                if method == "recency" and half_life_rows is not None
                else {}
            )
        ),
    }, accepted


def select_score_model(
    development_oos: pd.DataFrame,
    *,
    min_blocks: int = 3,
    min_relative_improvement: float = 0.005,
    max_metric_regression: float = 0.01,
    min_improvement_share: float = 2.0 / 3.0,
) -> dict[str, Any]:
    """Select a score method from development OOS only; locked OOS is never inspected."""
    if development_oos.empty:
        return {
            "selected_method": "primary",
            "status": "HOLD",
            "reason": "No development OOS rows available",
            "evaluated_blocks": 0,
        }

    work = development_oos.copy()
    primary = _summary(work, "primary")
    if len(work) < int(min_blocks):
        return {
            "selected_method": "primary",
            "status": "HOLD",
            "reason": f"Need at least {min_blocks} development OOS blocks for challenger selection",
            "evaluated_blocks": int(len(work)),
            "candidate_records": {"primary": {"status": "PRIMARY", "metrics": primary}},
        }
    if not np.isfinite(primary["score_logloss"]):
        return {
            "selected_method": "primary",
            "status": "HOLD",
            "reason": "Primary score LogLoss is non-finite",
            "evaluated_blocks": int(len(work)),
        }

    records: dict[str, dict[str, Any]] = {
        "primary": {"status": "PRIMARY", "metrics": primary}
    }
    selected = "primary"
    selected_loss = primary["score_logloss"]
    selected_parameters: dict[str, Any] = {}

    for key, half_life_days in _recency_specs(work):
        try:
            record, accepted = _check_challenger(
                work,
                primary,
                "recency",
                half_life_days=half_life_days,
                half_life_rows=None,
                max_metric_regression=max_metric_regression,
                min_relative_improvement=min_relative_improvement,
                min_improvement_share=min_improvement_share,
            )
        except (KeyError, ValueError) as exc:
            record, accepted = (
                {"status": "UNAVAILABLE", "reason": f"{type(exc).__name__}: {exc}", "metrics": {}},
                False,
            )
        records[key] = record
        if accepted and record["metrics"]["score_logloss"] < selected_loss:
            selected = "recency"
            selected_loss = record["metrics"]["score_logloss"]
            selected_parameters = dict(record.get("parameters") or {})

    if "dc_status" not in work.columns or not work["dc_status"].astype(str).eq("PASS").all():
        records["dixon_coles"] = {"status": "UNAVAILABLE", "metrics": {}}
    else:
        try:
            record, accepted = _check_challenger(
                work,
                primary,
                "dixon_coles",
                max_metric_regression=max_metric_regression,
                min_relative_improvement=min_relative_improvement,
                min_improvement_share=min_improvement_share,
            ) if {_col("dixon_coles", metric) for metric in METRICS}.issubset(work.columns) else (
                {"status": "UNAVAILABLE", "reason": "Missing Dixon-Coles development metrics", "metrics": {}},
                False,
            )
            records["dixon_coles"] = record
            if accepted and record["metrics"]["score_logloss"] < selected_loss:
                selected = "dixon_coles"
                selected_loss = record["metrics"]["score_logloss"]
                selected_parameters = {}
        except (KeyError, ValueError) as exc:
            records["dixon_coles"] = {
                "status": "UNAVAILABLE",
                "reason": f"{type(exc).__name__}: {exc}",
                "metrics": {},
            }

    return {
        "selected_method": selected,
        "selected_parameters": selected_parameters,
        "status": "ADOPT_CANDIDATE" if selected != "primary" else "KEEP_PRIMARY",
        "reason": (
            "Development-OOS challenger gate passed"
            if selected != "primary"
            else "No score challenger passed the development-OOS gate"
        ),
        "evaluated_blocks": int(len(work)),
        "candidate_records": records,
        "selection_rule": {
            "development_oos_only": True,
            "locked_oos_inspected": False,
            "min_development_blocks": int(min_blocks),
            "min_relative_improvement": min_relative_improvement,
            "max_metric_regression": max_metric_regression,
            "min_improvement_share": min_improvement_share,
            "recency_half_lives_days": [float(x) for x in RECENCY_HALF_LIVES_DAYS],
        },
    }


def verify_selected_score_model(
    selection: dict[str, Any],
    locked_oos: pd.DataFrame,
    *,
    max_metric_regression: float = 0.02,
    min_locked_blocks: int = 2,
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
    if len(locked_oos) < int(min_locked_blocks):
        return {
            "selected_method": "primary",
            "status": "HOLD",
            "reason": f"Need at least {min_locked_blocks} locked OOS blocks",
            "locked_oos_inspected": True,
            "locked_oos_blocks": int(len(locked_oos)),
        }

    base = _summary(locked_oos, "primary")
    required_finite = all(np.isfinite(base[m]) for m in METRICS)
    rows = int(
        pd.to_numeric(locked_oos["n"], errors="coerce").fillna(0).sum()
    ) if "n" in locked_oos.columns else 0
    market_evidence = {
        "score_logloss_finite": bool(np.isfinite(base["score_logloss"])),
        "over_2_5_logloss_finite": bool(np.isfinite(base["over_2_5_logloss"])),
        "over_2_5_brier_finite": bool(np.isfinite(base["over_2_5_brier"])),
        "btts_logloss_finite": bool(np.isfinite(base["btts_logloss"])),
        "btts_brier_finite": bool(np.isfinite(base["btts_brier"])),
    }
    market_metrics_finite = bool(all(market_evidence.values()) and rows > 0)

    if selected == "primary":
        return {
            "selected_method": "primary",
            "status": "PASS" if required_finite and market_metrics_finite else "REJECT",
            "reason": (
                "Primary score method retained with explicit finite locked score/O-U/BTTS evidence"
                if required_finite and market_metrics_finite
                else "Primary score locked evidence is incomplete or non-finite"
            ),
            "locked_oos_inspected": True,
            "locked_oos_blocks": int(len(locked_oos)),
            "locked_oos_rows": rows,
            "baseline_metrics": base,
            "selected_metrics": base,
            "market_evidence": market_evidence,
            "market_metrics_finite": market_metrics_finite,
            "checks": {
                "all_primary_score_metrics_finite": required_finite,
                "locked_oos_rows_positive": rows > 0,
                "market_metrics_finite": market_metrics_finite,
            },
        }

    if selected not in METHODS[1:]:
        return {
            "selected_method": "primary",
            "status": "REJECT",
            "reason": f"Unknown selected score method: {selected}",
            "locked_oos_inspected": True,
        }

    half_life_days = None
    half_life_rows = None
    if selected == "recency":
        params = selection.get("selected_parameters") or {}
        if "half_life_days" in params:
            half_life_days = float(params["half_life_days"])
        elif "half_life_rows" in params:
            half_life_rows = float(params["half_life_rows"])
        else:
            half_life_days = 365.0

    candidate_col_missing = {
        _col(
            selected,
            metric,
            half_life_days=half_life_days,
            half_life_rows=half_life_rows,
        )
        for metric in METRICS
    }.difference(locked_oos.columns)
    if candidate_col_missing and selected == "recency" and half_life_rows is None:
        # Read-only compatibility with old 800-row artifacts.
        legacy = {
            _col("recency", metric, half_life_rows=800.0)
            for metric in METRICS
        }
        if legacy.issubset(locked_oos.columns):
            half_life_rows = 800.0
            candidate_col_missing = set()

    if candidate_col_missing:
        return {
            "selected_method": "primary",
            "status": "REJECT",
            "reason": "Locked OOS score metrics are incomplete",
            "locked_oos_inspected": True,
            "locked_oos_blocks": int(len(locked_oos)),
            "locked_oos_rows": rows,
        }

    if selected == "recency":
        status_col = (
            f"recency_d{int(half_life_days)}_status"
            if half_life_days is not None
            else "recency_h800_status"
        )
        if status_col not in locked_oos.columns:
            status_col = "recency_status"
    else:
        status_col = "dc_status"

    if status_col in locked_oos.columns and not locked_oos[status_col].astype(str).eq("PASS").all():
        return {
            "selected_method": "primary",
            "status": "REJECT",
            "reason": f"{selected} is unavailable on locked OOS",
            "locked_oos_inspected": True,
            "locked_oos_blocks": int(len(locked_oos)),
            "locked_oos_rows": rows,
        }

    try:
        selected_metrics = _summary(
            locked_oos,
            selected,
            half_life_days=half_life_days,
            half_life_rows=half_life_rows,
        )
    except KeyError:
        return {
            "selected_method": "primary",
            "status": "REJECT",
            "reason": "Locked OOS selected-model metrics are incomplete",
            "locked_oos_inspected": True,
            "locked_oos_blocks": int(len(locked_oos)),
            "locked_oos_rows": rows,
        }

    if not all(np.isfinite(base[m]) and np.isfinite(selected_metrics[m]) for m in METRICS):
        return {
            "selected_method": "primary",
            "status": "REJECT",
            "reason": "Non-finite locked OOS metric",
            "locked_oos_inspected": True,
            "locked_oos_blocks": int(len(locked_oos)),
            "locked_oos_rows": rows,
            "market_evidence": market_evidence,
            "market_metrics_finite": market_metrics_finite,
        }

    relative_deltas = {
        m: float((selected_metrics[m] - base[m]) / max(abs(base[m]), 1e-9))
        for m in METRICS
    }
    selected_ll_col = _col(
        selected,
        "score_logloss",
        half_life_days=half_life_days,
        half_life_rows=half_life_rows,
    )
    block_ok = []
    for _, row in locked_oos.iterrows():
        bp = pd.to_numeric(row["score_logloss"], errors="coerce")
        cp = pd.to_numeric(row[selected_ll_col], errors="coerce")
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
    passed = bool(
        overall_score_ok
        and secondary_ok
        and blocks_ok
        and market_metrics_finite
    )

    return {
        "selected_method": selected if passed else "primary",
        "status": "PASS" if passed else "REJECT",
        "reason": (
            "Selected challenger survived untouched locked OOS"
            if passed
            else "Selected challenger failed locked OOS verification"
        ),
        "locked_oos_inspected": True,
        "locked_oos_blocks": int(len(locked_oos)),
        "locked_oos_rows": rows,
        "baseline_metrics": base,
        "selected_metrics": selected_metrics,
        "relative_deltas": relative_deltas,
        "market_evidence": market_evidence,
        "market_metrics_finite": market_metrics_finite,
        "checks": {
            "overall_score_logloss_not_worse": overall_score_ok,
            "secondary_metrics_not_materially_worse": secondary_ok,
            "block_level_score_logloss_not_materially_worse": blocks_ok,
            "market_metrics_finite": market_metrics_finite,
        },
        **(
            {
                "selected_parameters": (
                    {"half_life_days": half_life_days}
                    if half_life_days is not None
                    else {"half_life_rows": half_life_rows}
                )
            }
            if selected == "recency"
            else {}
        ),
    }
