from __future__ import annotations

"""Fail-closed health audit for the daily soccer forecast artifact."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.prediction.research_forecast import FORECAST_COLUMNS


PREDICTED_STATUSES = {
    "PREDICTED_PRODUCTION_ADOPTED",
    "PREDICTED_FALLBACK_BASELINE",
    "PREDICTED_FALLBACK_BASELINE_AFTER_PRODUCTION_DEFER",
}
NO_TARGET_STATUSES = {"NO_TARGET_FIXTURES"}
BLOCKED_STATUSES = {
    "DEFERRED_NO_ADOPTED_MODEL",
    "DEFERRED_NO_PIT_ELIGIBLE_FIXTURES",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _utc_timestamp(value: Any, field: str) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(value)
    except Exception as exc:
        raise RuntimeError(f"invalid {field}") from exc
    if ts.tzinfo is None:
        raise RuntimeError(f"{field} must include timezone")
    return ts.tz_convert("UTC")


def _probability_checks(df: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    for idx, row in df.iterrows():
        p = np.asarray([
            float(row["home_win_probability"]),
            float(row["draw_probability"]),
            float(row["away_win_probability"]),
        ])
        if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
            failures.append(f"{idx}: invalid 1X2 probabilities")
        elif not np.isclose(float(p.sum()), 1.0, atol=1e-6):
            failures.append(f"{idx}: 1X2 probabilities do not sum to 1")

        result_prob = float(row["result_prediction_probability"])
        if not np.isfinite(result_prob) or not 0.0 <= result_prob <= 1.0:
            failures.append(f"{idx}: invalid result_prediction_probability")
        elif not np.isclose(result_prob, float(p.max()), atol=1e-6):
            failures.append(f"{idx}: result_prediction_probability mismatch")

        for rank in (1, 2, 3):
            score = str(row[f"score_{rank}"]).strip()
            sp = float(row[f"score_{rank}_probability"])
            if not score or "-" not in score:
                failures.append(f"{idx}: invalid score_{rank}")
            if not np.isfinite(sp) or not 0.0 <= sp <= 1.0:
                failures.append(f"{idx}: invalid score_{rank}_probability")

        ou = np.asarray([
            float(row["over_2_5_probability"]),
            float(row["under_2_5_probability"]),
        ])
        if not np.isfinite(ou).all() or np.any((ou < 0.0) | (ou > 1.0)) or not np.isclose(ou.sum(), 1.0, atol=1e-6):
            failures.append(f"{idx}: invalid O/U 2.5 probabilities")

        btts = np.asarray([
            float(row["btts_yes_probability"]),
            float(row["btts_no_probability"]),
        ])
        if not np.isfinite(btts).all() or np.any((btts < 0.0) | (btts > 1.0)) or not np.isclose(btts.sum(), 1.0, atol=1e-6):
            failures.append(f"{idx}: invalid BTTS probabilities")

        status = str(row.get("mom_status", ""))
        if status.startswith("PREDICTED"):
            names = [str(row[f"mom_{rank}_player"]).strip() for rank in (1, 2, 3, 4)]
            mp = np.asarray([float(row[f"mom_{rank}_probability"]) for rank in (1, 2, 3, 4)])
            if any(not name for name in names) or len(set(names)) != 4:
                failures.append(f"{idx}: invalid MOM top4 identity")
            if not np.isfinite(mp).all() or np.any(mp < 0.0) or not np.isclose(mp.sum(), 1.0, atol=1e-6):
                failures.append(f"{idx}: invalid MOM top4 probabilities")

    return failures


def audit(output_path: str, status_path: str) -> dict[str, Any]:
    output = Path(output_path)
    status_file = Path(status_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "BLOCKED",
        "ok": False,
        "forecast_evidence": "NONE",
        "prediction_runtime_ready": False,
        "failures": [],
    }

    status = _load_json(status_file)
    if not status:
        report["failures"] = ["missing_or_invalid_status"]
        return report

    forecast_status = str(status.get("status", "")).strip().upper()
    report["source_status"] = forecast_status
    report["target_rows"] = int(status.get("rows", 0) or 0)
    report["prediction_rows"] = int(status.get("prediction_rows", 0) or 0)

    try:
        prediction_time = _utc_timestamp(status.get("prediction_time_utc"), "prediction_time_utc")
    except RuntimeError as exc:
        report["failures"] = [str(exc)]
        return report
    report["prediction_time_utc"] = prediction_time.isoformat()

    if not output.is_file() or output.stat().st_size <= 0:
        report["failures"] = ["missing_or_empty_forecast_csv"]
        return report

    try:
        df = pd.read_csv(output)
    except Exception as exc:
        report["failures"] = [f"unreadable_forecast_csv:{type(exc).__name__}"]
        return report

    report["output_rows"] = int(len(df))
    missing = sorted(set(FORECAST_COLUMNS) - set(df.columns))
    if missing:
        report["failures"] = [f"missing_forecast_columns:{missing}"]
        return report

    if forecast_status in NO_TARGET_STATUSES:
        failures: list[str] = []
        if not df.empty:
            failures.append("NO_TARGET_FIXTURES requires empty forecast output")
        if report["target_rows"] != 0 or report["prediction_rows"] != 0:
            failures.append("NO_TARGET_FIXTURES row counts must be zero")
        report["status"] = "NO_TARGET"
        report["forecast_evidence"] = "NONE"
        report["ok"] = not failures
        report["prediction_runtime_ready"] = False
        report["failures"] = failures
        return report

    if forecast_status in BLOCKED_STATUSES:
        report["status"] = "BLOCKED"
        report["failures"] = [f"upstream_status:{forecast_status}"]
        return report

    if forecast_status not in PREDICTED_STATUSES:
        report["failures"] = [f"unknown_forecast_status:{forecast_status}"]
        return report

    failures = []
    if df.empty:
        failures.append("predicted_status_requires_nonempty_forecast")
    if report["prediction_rows"] != len(df):
        failures.append("status_prediction_rows_mismatch")
    if report["target_rows"] <= 0:
        failures.append("predicted_status_requires_positive_target_rows")
    if report["prediction_rows"] <= 0:
        failures.append("predicted_status_requires_positive_prediction_rows")
    fallback_status = forecast_status.startswith("PREDICTED_FALLBACK_BASELINE")
    if fallback_status:
        if status.get("production_model_used") is True:
            failures.append("fallback_status_cannot_claim_production_model")
        if status.get("fallback_used") is not True:
            failures.append("fallback_status_requires_fallback_used")
        if status.get("research_only") is not True:
            failures.append("fallback_status_must_be_research_only")
        if forecast_status == "PREDICTED_FALLBACK_BASELINE_AFTER_PRODUCTION_DEFER":
            if not isinstance(status.get("runner_status"), dict):
                failures.append("production_defer_fallback_requires_runner_status")
    else:
        if status.get("production_model_used") is not True:
            failures.append("predicted_production_status_requires_production_model")
        if status.get("research_heuristic_disabled") is not True:
            failures.append("heuristic_prediction_must_remain_disabled")

        runner_status = status.get("runner_status")
        if not isinstance(runner_status, dict):
            failures.append("predicted_status_requires_runner_status_evidence")
        else:
            if str(runner_status.get("status", "")).upper() != "PREDICTED":
                failures.append("runner_status_must_be_PREDICTED")
            freshness = runner_status.get("freshness")
            if not isinstance(freshness, dict) or str(freshness.get("status", "")).upper() != "FRESH":
                failures.append("prediction_requires_FRESH_matchday_snapshot")

    if not df.empty:
        ids = df["match_id"].astype(str).str.strip()
        if ids.eq("").any():
            failures.append("empty_match_id")
        if ids.duplicated().any():
            failures.append("duplicate_match_id")

        kickoff = pd.to_datetime(df["kickoff_utc"], utc=True, errors="coerce")
        if kickoff.isna().any():
            failures.append("invalid_kickoff_utc")
        elif not bool((kickoff > prediction_time).all()):
            failures.append("prediction_contains_non_future_kickoff")

        if df["home_team"].astype(str).str.strip().eq("").any():
            failures.append("empty_home_team")
        if df["away_team"].astype(str).str.strip().eq("").any():
            failures.append("empty_away_team")
        failures.extend(_probability_checks(df))

    report["status"] = "HEALTHY" if not failures else "BLOCKED"
    report["ok"] = not failures
    report["forecast_evidence"] = "PREDICTION_VERIFIED" if not failures else "PREDICTION_INVALID"
    report["prediction_runtime_ready"] = not failures
    report["coverage_ratio"] = (
        report["prediction_rows"] / report["target_rows"]
        if report["target_rows"] > 0 else 0.0
    )
    report["failures"] = failures[:50]
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--report", default="artifacts/daily_research_forecast_health.json")
    args = parser.parse_args()

    result = audit(args.output, args.status)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if bool(result.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
