from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.prediction.model_bundle import load_bundle, predict_bundle


REQUIRED_FIXTURE_COLUMNS = {
    "match_id",
    "kickoff_utc",
    "home_team",
    "away_team",
    "source_available_at_utc",
    "pit_verified",
    "starter_status",
}


def load_adopted_model(registry_path: str = "artifacts/model_registry.json") -> dict:
    p = Path(registry_path)
    if not p.exists():
        raise RuntimeError("No adopted model registry exists; production prediction is fail-closed")
    record = json.loads(p.read_text(encoding="utf-8"))
    if record.get("adoption_status") != "ADOPT":
        raise RuntimeError("Registry contains no ADOPT model")
    return record


def _write_status(path: Path, status: str, **extra: object) -> dict:
    payload = {"status": status, **extra}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return payload


def _eligible_fixtures(fixtures: pd.DataFrame, prediction_time: pd.Timestamp) -> pd.DataFrame:
    missing = sorted(REQUIRED_FIXTURE_COLUMNS - set(fixtures.columns))
    if missing:
        raise RuntimeError(f"Future fixture input missing required columns: {missing}")
    d = fixtures.copy()
    d["kickoff_utc"] = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    d["source_available_at_utc"] = pd.to_datetime(d["source_available_at_utc"], utc=True, errors="coerce")
    d["pit_verified"] = d["pit_verified"].astype(bool)
    d["starter_status"] = d["starter_status"].astype(str).str.upper().str.strip()
    d = d.dropna(subset=["kickoff_utc", "source_available_at_utc"])
    # Production is strictly forward-looking: never use a fixture or information that
    # was not available by the exact prediction timestamp.
    d = d[
        (d["kickoff_utc"] > prediction_time)
        & (d["source_available_at_utc"] <= prediction_time)
        & d["pit_verified"]
        & d["starter_status"].isin({"ANNOUNCED", "CONFIRMED"})
    ].copy()
    return d.drop_duplicates(subset=["match_id"], keep="last").sort_values("kickoff_utc", kind="mergesort")


def run(
    fixtures_path: str = "artifacts/future_fixtures.csv",
    bundle_path: str = "artifacts/production_model.pkl",
    output_path: str = "artifacts/predictions.csv",
    status_path: str = "artifacts/prediction_status.json",
    prediction_time: str | None = None,
) -> dict:
    status_file = Path(status_path)
    now = pd.Timestamp(prediction_time, tz="UTC") if prediction_time else pd.Timestamp(datetime.now(timezone.utc))
    load_adopted_model()
    bundle = load_bundle(bundle_path)
    p = Path(fixtures_path)
    if not p.exists():
        return _write_status(status_file, "NO_FIXTURE_INPUT", prediction_time_utc=now.isoformat(), oos_claimed=False)

    fixtures = pd.read_csv(p)
    eligible = _eligible_fixtures(fixtures, now)
    if eligible.empty:
        return _write_status(
            status_file,
            "NO_ELIGIBLE_FIXTURES",
            prediction_time_utc=now.isoformat(),
            source_rows=int(len(fixtures)),
            eligible_rows=0,
            oos_claimed=False,
        )

    probs = predict_bundle(bundle, eligible)
    if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("Production prediction produced invalid probabilities")
    result = eligible[["match_id", "kickoff_utc", "home_team", "away_team"]].copy()
    result["p_home"] = probs[:, 0]
    result["p_draw"] = probs[:, 1]
    result["p_away"] = probs[:, 2]
    labels = np.array(["H", "D", "A"])
    result["prediction"] = labels[np.argmax(probs, axis=1)]
    result["confidence"] = probs.max(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    result["margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    result["abstain"] = (result["confidence"] < 0.45) | (result["margin"] < 0.08)
    result["prediction_time_utc"] = now.isoformat()
    result["model_version"] = str(bundle["model_version"])
    result.to_csv(output_path, index=False)
    return _write_status(
        status_file,
        "PREDICTED",
        prediction_time_utc=now.isoformat(),
        source_rows=int(len(fixtures)),
        eligible_rows=int(len(eligible)),
        prediction_rows=int(len(result)),
        abstained_rows=int(result["abstain"].sum()),
        output_path=str(output_path),
        model_version=str(bundle["model_version"]),
        oos_claimed=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", default="artifacts/future_fixtures.csv")
    parser.add_argument("--bundle", default="artifacts/production_model.pkl")
    parser.add_argument("--output", default="artifacts/predictions.csv")
    parser.add_argument("--status", default="artifacts/prediction_status.json")
    parser.add_argument("--prediction-time", default=None)
    args = parser.parse_args()
    result = run(args.fixtures, args.bundle, args.output, args.status, args.prediction_time)
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
