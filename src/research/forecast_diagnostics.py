from __future__ import annotations

"""Generate outcome-free per-fixture uncertainty telemetry for research."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.prediction.model_bundle import load_bundle, predict_bundle_with_diagnostics

OUTPUT_COLUMNS = (
    "match_id", "kickoff_utc", "competition", "home_team", "away_team",
    "p_home", "p_draw", "p_away", "confidence", "margin",
    "model_disagreement", "predictive_entropy", "uncertainty_score",
    "covariate_drift", "history_support_risk", "routing_risk",
    "routing_risk_bucket", "routing_route",
)


def build(fixtures: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    required = {"match_id", "kickoff_utc", "competition", "home_team", "away_team"}
    missing = sorted(required - set(fixtures.columns))
    if missing:
        raise RuntimeError(f"diagnostic fixture input missing columns: {missing}")

    kickoff = pd.to_datetime(fixtures["kickoff_utc"], utc=True, errors="coerce")
    if kickoff.isna().any():
        raise RuntimeError("diagnostic input contains invalid kickoff_utc values")
    match_ids = fixtures["match_id"].astype(str).str.strip()
    if match_ids.eq("").any():
        raise RuntimeError("diagnostic input contains empty match_id values")

    probs, diagnostics = predict_bundle_with_diagnostics(bundle, fixtures)
    probs = np.asarray(probs, dtype=float)
    if probs.shape != (len(fixtures), 3) or not np.isfinite(probs).all():
        raise RuntimeError("diagnostic prediction probabilities are invalid")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("diagnostic prediction probabilities are not normalized")

    sorted_probs = np.sort(probs, axis=1)
    risk = np.asarray(diagnostics["routing_risk"], dtype=float)
    buckets = np.where(risk < 0.33, "LOW", np.where(risk < 0.66, "MEDIUM", "HIGH"))

    result = pd.DataFrame(
        {
            "match_id": match_ids,
            "kickoff_utc": kickoff.astype(str),
            "competition": fixtures["competition"].astype(str),
            "home_team": fixtures["home_team"].astype(str),
            "away_team": fixtures["away_team"].astype(str),
            "p_home": probs[:, 0],
            "p_draw": probs[:, 1],
            "p_away": probs[:, 2],
            "confidence": probs.max(axis=1),
            "margin": sorted_probs[:, -1] - sorted_probs[:, -2],
            "model_disagreement": diagnostics["model_disagreement"],
            "predictive_entropy": diagnostics["predictive_entropy"],
            "uncertainty_score": diagnostics["uncertainty_score"],
            "covariate_drift": diagnostics["covariate_drift"],
            "history_support_risk": diagnostics["history_support_risk"],
            "routing_risk": risk,
            "routing_risk_bucket": buckets,
            "routing_route": diagnostics["route"],
        },
        columns=OUTPUT_COLUMNS,
    )
    if result["match_id"].duplicated().any():
        raise RuntimeError("diagnostic output contains duplicate match_id values")
    numeric = list(OUTPUT_COLUMNS[5:16])
    values = result[numeric].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise RuntimeError("diagnostic output contains unbounded numeric values")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", default="artifacts/forecast_uncertainty_shadow.csv")
    parser.add_argument("--status", default="artifacts/forecast_uncertainty_shadow_status.json")
    args = parser.parse_args()

    result = build(pd.read_csv(args.fixtures), load_bundle(args.bundle))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    status = {
        "status": "SHADOW_TELEMETRY_WRITTEN",
        "rows": int(len(result)),
        "outcome_free": True,
        "production_prediction_unchanged": True,
        "output": str(output),
    }
    status_path = Path(args.status)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
