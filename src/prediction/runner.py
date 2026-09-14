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
    try:
        record = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Adopted model registry is unreadable: {type(exc).__name__}: {exc}") from exc
    if not isinstance(record, dict):
        raise RuntimeError("Adopted model registry must contain a JSON object")
    if record.get("adoption_status") != "ADOPT":
        raise RuntimeError("Registry contains no ADOPT model")
    return record


def _write_status(path: Path, status: str, **extra: object) -> dict:
    payload = {"status": status, **extra}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return payload


def _strict_bool(series: pd.Series, name: str) -> pd.Series:
    """Parse booleans without allowing strings/NaN to silently become True."""
    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise RuntimeError(f"Future fixture column {name!r} contains missing boolean values")
        return series
    normalized = series.astype("string").str.strip().str.lower()
    valid = normalized.isin({"true", "false", "1", "0", "yes", "no"})
    if not valid.all():
        bad = sorted(normalized[~valid].dropna().unique().tolist())[:10]
        raise RuntimeError(f"Future fixture column {name!r} contains non-boolean values: {bad}")
    return normalized.isin({"true", "1", "yes"})


def _normalize_prediction_time(value: str | None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp(datetime.now(timezone.utc))
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _eligible_fixtures(fixtures: pd.DataFrame, prediction_time: pd.Timestamp) -> pd.DataFrame:
    missing = sorted(REQUIRED_FIXTURE_COLUMNS - set(fixtures.columns))
    if missing:
        raise RuntimeError(f"Future fixture input missing required columns: {missing}")
    d = fixtures.copy()
    d["match_id"] = d["match_id"].astype("string").str.strip()
    if d["match_id"].isna().any() or d["match_id"].eq("").any():
        raise RuntimeError("Future fixture input contains missing/empty match_id values")
    duplicate_ids = d.loc[d["match_id"].duplicated(keep=False), "match_id"].dropna().unique().tolist()
    if duplicate_ids:
        sample = sorted(map(str, duplicate_ids))[:10]
        raise RuntimeError(f"Future fixture input contains duplicate match_id values; refusing ambiguous prediction: {sample}")
    for team_col in ("home_team", "away_team"):
        d[team_col] = d[team_col].astype("string").str.strip()
        if d[team_col].isna().any() or d[team_col].eq("").any():
            raise RuntimeError(f"Future fixture input contains missing/empty {team_col} values")
    d["kickoff_utc"] = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    d["source_available_at_utc"] = pd.to_datetime(d["source_available_at_utc"], utc=True, errors="coerce")
    if d["kickoff_utc"].isna().any():
        raise RuntimeError("Future fixture input contains invalid/missing kickoff_utc values")
    if d["source_available_at_utc"].isna().any():
        raise RuntimeError("Future fixture input contains invalid/missing source_available_at_utc values")
    d["pit_verified"] = _strict_bool(d["pit_verified"], "pit_verified")
    d["starter_status"] = d["starter_status"].astype("string").str.upper().str.strip()
    if d["starter_status"].isna().any() or d["starter_status"].eq("").any():
        raise RuntimeError("Future fixture input contains missing/empty starter_status values")
    # Production is strictly forward-looking: never use a fixture or information that
    # was not available by the exact prediction timestamp. Invalid input is a hard error,
    # not a silently dropped row, so upstream data regressions cannot hide behind a green run.
    d = d[
        (d["kickoff_utc"] > prediction_time)
        & (d["source_available_at_utc"] <= prediction_time)
        & d["pit_verified"]
        & d["starter_status"].isin({"ANNOUNCED", "CONFIRMED"})
    ].copy()
    return d.sort_values("kickoff_utc", kind="mergesort")


def run(
    fixtures_path: str = "artifacts/future_fixtures.csv",
    bundle_path: str = "artifacts/production_model.pkl",
    output_path: str = "artifacts/predictions.csv",
    status_path: str = "artifacts/prediction_status.json",
    prediction_time: str | None = None,
    registry_path: str = "artifacts/model_registry.json",
) -> dict:
    status_file = Path(status_path)
    now = _normalize_prediction_time(prediction_time)
    registry = load_adopted_model(registry_path)
    bundle = load_bundle(bundle_path)
    registry_version = registry.get("model_version")
    bundle_version = bundle.get("model_version")
    if registry_version is not None and str(registry_version) != str(bundle_version):
        raise RuntimeError(
            f"Adopted registry/model bundle version mismatch: registry={registry_version!r}, bundle={bundle_version!r}"
        )
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
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_file.with_suffix(output_file.suffix + ".tmp")
    result.to_csv(temp_output, index=False)
    temp_output.replace(output_file)
    return _write_status(
        status_file,
        "PREDICTED",
        prediction_time_utc=now.isoformat(),
        source_rows=int(len(fixtures)),
        eligible_rows=int(len(eligible)),
        prediction_rows=int(len(result)),
        abstained_rows=int(result["abstain"].sum()),
        output_path=str(output_file),
        model_version=str(bundle["model_version"]),
        oos_claimed=bool(registry.get("oos_verified", False)),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", default="artifacts/future_fixtures.csv")
    parser.add_argument("--bundle", default="artifacts/production_model.pkl")
    parser.add_argument("--output", default="artifacts/predictions.csv")
    parser.add_argument("--status", default="artifacts/prediction_status.json")
    parser.add_argument("--prediction-time", default=None)
    parser.add_argument("--registry", default="artifacts/model_registry.json")
    args = parser.parse_args()
    result = run(args.fixtures, args.bundle, args.output, args.status, args.prediction_time, args.registry)
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
