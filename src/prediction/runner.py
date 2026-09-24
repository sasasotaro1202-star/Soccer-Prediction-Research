from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.prediction.model_bundle import load_bundle, predict_bundle
from src.prediction.active_model import resolve_active_production_paths, resolve_best_available_paths
from src.prediction.secondary_outputs import predict_mom_candidates, predict_score_candidates, predict_score_markets
from src.prediction.matchday_intelligence import apply_matchday_intelligence


REQUIRED_FIXTURE_COLUMNS = {
    "match_id",
    "kickoff_utc",
    "home_team",
    "away_team",
    "competition",
    "source_available_at_utc",
    "pit_verified",
    "starter_status",
}

# These are intentionally presentation/abstention defaults, not claims about model quality.
# They should eventually be calibrated from chronological OOS data and stored with the adopted model.
ABSTAIN_CONFIDENCE_THRESHOLD = 0.45
ABSTAIN_MARGIN_THRESHOLD = 0.08


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


def load_best_available_model(registry_path: str) -> dict:
    p = Path(registry_path)
    if not p.exists():
        raise RuntimeError("Selected model registry does not exist; no safe prediction model is available")
    try:
        record = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Selected model registry is unreadable: {type(exc).__name__}: {exc}") from exc
    if not isinstance(record, dict):
        raise RuntimeError("Selected model registry must contain a JSON object")
    status = str(record.get("adoption_status", "")).upper()
    if status not in {"ADOPT", "VALIDATED_CANDIDATE"}:
        raise RuntimeError(f"Selected registry status is not prediction-safe: {status}")
    return record


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def _write_status(path: Path, status: str, **extra: object) -> dict:
    payload = {"status": status, **extra}
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_production_provenance(bundle_path: str, registry: dict, bundle: dict) -> None:
    """Require exact production artifact provenance before serving schema>=2 models."""
    root = Path(bundle_path).parent
    provenance_path = root / "production_provenance.json"
    if not provenance_path.exists():
        raise RuntimeError("Production provenance is missing; refusing schema>=2 prediction")

    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Production provenance is unreadable: {type(exc).__name__}: {exc}") from exc
    if not isinstance(provenance, dict):
        raise RuntimeError("Production provenance must be a JSON object")

    recorded_version = provenance.get("production_model_json_version")
    if recorded_version is not None and str(recorded_version) != str(bundle.get("model_version")):
        raise RuntimeError("Production provenance model version mismatch")
    recorded_registry_version = provenance.get("registry_model_version")
    if recorded_registry_version is not None and str(recorded_registry_version) != str(registry.get("model_version")):
        raise RuntimeError("Production provenance registry version mismatch")

    files = provenance.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("Production provenance file hashes are missing")

    for name, path in (
        ("production_model.pkl", Path(bundle_path)),
        ("production_model.json", root / "production_model.json"),
        ("model_registry.json", root / "model_registry.json"),
    ):
        entry = files.get(name)
        expected = entry.get("sha256") if isinstance(entry, dict) else None
        if not isinstance(expected, str) or len(expected) != 64:
            raise RuntimeError(f"Production provenance hash missing for {name}")
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Production provenance artifact missing: {name}")
        if _sha256(path) != expected:
            raise RuntimeError(f"Production provenance hash mismatch: {name}")


def _verify_validated_candidate_provenance(bundle_path: str, registry: dict, bundle: dict) -> None:
    root = Path(bundle_path).parent
    provenance_path = root / "validated_candidate_provenance.json"
    if not provenance_path.exists():
        raise RuntimeError("Validated candidate provenance is missing")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Validated candidate provenance is unreadable: {type(exc).__name__}: {exc}") from exc
    if provenance.get("status") != "VALIDATED_CANDIDATE":
        raise RuntimeError("Validated candidate provenance status is invalid")
    if provenance.get("locked_oos_used_for_training") is not False:
        raise RuntimeError("Validated candidate provenance indicates locked-OOS training contamination")
    if str(provenance.get("model_version")) != str(bundle.get("model_version")):
        raise RuntimeError("Validated candidate model version mismatch")
    if str(provenance.get("model_version")) != str(registry.get("model_version")):
        raise RuntimeError("Validated candidate registry/model version mismatch")
    gate = provenance.get("calibration_gate")
    if not isinstance(gate, dict) or gate.get("status") != "PASS":
        raise RuntimeError("Validated candidate calibration gate is not PASS")
    files = provenance.get("files")
    required = {
        "validated_candidate_model.pkl": Path(bundle_path),
        "validated_candidate_model.json": root / "validated_candidate_model.json",
        "validated_candidate_registry.json": root / "validated_candidate_registry.json",
    }
    if not isinstance(files, dict):
        raise RuntimeError("Validated candidate provenance hashes are missing")
    for name, path in required.items():
        entry = files.get(name)
        expected = entry.get("sha256") if isinstance(entry, dict) else None
        if not isinstance(expected, str) or len(expected) != 64:
            raise RuntimeError(f"Validated candidate provenance hash missing for {name}")
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Validated candidate artifact missing: {name}")
        if _sha256(path) != expected:
            raise RuntimeError(f"Validated candidate provenance hash mismatch: {name}")


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
    d = d[
        (d["kickoff_utc"] > prediction_time)
        & (d["source_available_at_utc"] <= prediction_time)
        & d["pit_verified"]
        & d["starter_status"].isin({"ANNOUNCED", "CONFIRMED"})
    ].copy()
    return d.sort_values("kickoff_utc", kind="mergesort")


def _merge_matchday_snapshot(
    fixtures: pd.DataFrame,
    prediction_time: pd.Timestamp,
    snapshot_path: str = "artifacts/future_matchday_fixtures.csv",
) -> tuple[pd.DataFrame, str]:
    """Overlay a recent PIT-verified matchday snapshot onto stable fixture rows."""
    path = Path(snapshot_path)
    if not path.is_file() or path.stat().st_size <= 0:
        return fixtures, "ABSENT"

    snapshot = pd.read_csv(path)
    required = {
        "match_id",
        "matchday_available_at_utc",
        "matchday_pit_verified",
        "matchday_source",
    }
    missing = sorted(required - set(snapshot.columns))
    if missing:
        return fixtures, "BLOCKED_SCHEMA:" + ",".join(missing)

    snapshot["match_id"] = snapshot["match_id"].astype("string").str.strip()
    if snapshot["match_id"].isna().any() or snapshot["match_id"].eq("").any():
        raise RuntimeError("Matchday snapshot contains missing/empty match_id values")
    if snapshot["match_id"].duplicated().any():
        raise RuntimeError("Matchday snapshot contains duplicate match_id values")

    snapshot["matchday_available_at_utc"] = pd.to_datetime(
        snapshot["matchday_available_at_utc"], utc=True, errors="coerce"
    )
    snapshot["matchday_pit_verified"] = _strict_bool(
        snapshot["matchday_pit_verified"], "matchday_pit_verified"
    )
    usable = snapshot[
        snapshot["matchday_available_at_utc"].notna()
        & (snapshot["matchday_available_at_utc"] <= prediction_time)
        & snapshot["matchday_pit_verified"]
    ].copy()
    if usable.empty:
        return fixtures, "NO_PIT_SAFE_ROWS"

    base = fixtures.copy()
    base["match_id"] = base["match_id"].astype("string").str.strip()
    if base["match_id"].duplicated().any():
        raise RuntimeError("Fixture input contains duplicate match_id values")

    overlay_cols = [
        c for c in usable.columns
        if c.startswith("matchday_") or c == "starter_status"
    ]
    overlay = usable[["match_id", *overlay_cols]].copy()

    if "starter_status" in overlay.columns and "starter_status" in base.columns:
        existing = base[["match_id", "starter_status"]]
        merged_status = overlay[["match_id", "starter_status"]].merge(
            existing, on="match_id", how="left", suffixes=("", "_existing")
        )
        confirmed = merged_status["starter_status"].astype("string").str.upper().isin(
            {"ANNOUNCED", "CONFIRMED"}
        )
        merged_status["starter_status"] = np.where(
            confirmed,
            merged_status["starter_status"],
            merged_status["starter_status_existing"],
        )
        overlay = overlay.drop(columns=["starter_status"]).merge(
            merged_status, on="match_id", how="left", validate="one_to_one"
        )

    base = base.merge(
        overlay,
        on="match_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_matchday"),
    )
    for column in [c for c in base.columns if c.endswith("_matchday")]:
        root = column[:-9]
        if root in base.columns:
            if root == "starter_status":
                promoted = base[column].astype("string").str.upper().isin(
                    {"ANNOUNCED", "CONFIRMED"}
                )
                base[root] = np.where(promoted, base[column], base[root])
            else:
                base[root] = base[column].combine_first(base[root])
            base = base.drop(columns=[column])
        else:
            base = base.rename(columns={column: root})
    return base, "APPLIED"


def run(
    fixtures_path: str = "artifacts/future_fixtures.csv",
    bundle_path: str = "artifacts/production_model.pkl",
    output_path: str = "artifacts/predictions.csv",
    status_path: str = "artifacts/prediction_status.json",
    prediction_time: str | None = None,
    registry_path: str = "artifacts/model_registry.json",
    model_policy: str = "best_available",
) -> dict:
    status_file = Path(status_path)
    now = _normalize_prediction_time(prediction_time)
    if model_policy not in {"production", "best_available"}:
        raise ValueError("model_policy must be production or best_available")
    model_mode = "PRODUCTION_ADOPTED"
    standard_paths = (
        bundle_path == "artifacts/production_model.pkl"
        and registry_path == "artifacts/model_registry.json"
    )
    if model_policy == "best_available" and standard_paths:
        bundle_path, registry_path, model_mode = resolve_best_available_paths()
        registry = load_best_available_model(registry_path)
    else:
        # Explicit caller paths are preserved. This keeps isolated validation runs
        # deterministic while the normal production entrypoint still uses best_available.
        bundle_path, registry_path = resolve_active_production_paths(bundle_path, registry_path)
        registry = load_adopted_model(registry_path)
    bundle = load_bundle(bundle_path)
    registry_version = registry.get("model_version")
    bundle_version = bundle.get("model_version")
    if registry_version is not None and str(registry_version) != str(bundle_version):
        raise RuntimeError(
            f"Adopted registry/model bundle version mismatch: registry={registry_version!r}, bundle={bundle_version!r}"
        )
    if int(bundle.get("schema_version", 1)) >= 2:
        if model_mode == "VALIDATED_CANDIDATE":
            _verify_validated_candidate_provenance(bundle_path, registry, bundle)
        else:
            _verify_production_provenance(bundle_path, registry, bundle)
    p = Path(fixtures_path)
    if not p.exists():
        return _write_status(status_file, "NO_FIXTURE_INPUT", prediction_time_utc=now.isoformat(), oos_claimed=False)

    fixtures = pd.read_csv(p)
    fixtures, matchday_merge_status = _merge_matchday_snapshot(fixtures, now)
    eligible = _eligible_fixtures(fixtures, now)
    if eligible.empty:
        return _write_status(
            status_file,
            "NO_ELIGIBLE_FIXTURES",
            prediction_time_utc=now.isoformat(),
            source_rows=int(len(fixtures)),
            eligible_rows=0,
            oos_claimed=False,
            matchday_merge_status=matchday_merge_status,
        )

    raw_probs = predict_bundle(bundle, eligible)
    try:
        probs = np.asarray(raw_probs, dtype=float)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Production prediction produced non-numeric probabilities") from exc
    if probs.shape != (len(eligible), 3):
        raise RuntimeError(f"Production prediction returned unexpected probability shape: {probs.shape}")
    if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("Production prediction produced invalid probabilities")
    probs, matchday_diagnostics = apply_matchday_intelligence(probs, eligible, now)
    if probs.shape != (len(eligible), 3) or not np.isfinite(probs).all():
        raise RuntimeError("Matchday intelligence produced invalid probabilities")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("Matchday intelligence produced non-normalized probabilities")
    result = eligible[["match_id", "kickoff_utc", "home_team", "away_team"]].copy()
    result["p_home"] = probs[:, 0]
    result["p_draw"] = probs[:, 1]
    result["p_away"] = probs[:, 2]
    labels = np.array(["H", "D", "A"])
    result["prediction"] = labels[np.argmax(probs, axis=1)]
    result["confidence"] = probs.max(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    result["margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    result["low_confidence"] = (result["confidence"] < ABSTAIN_CONFIDENCE_THRESHOLD) | (
        result["margin"] < ABSTAIN_MARGIN_THRESHOLD
    )
    # Keep every eligible fixture in the prediction output. low_confidence is a separate
    # classification, so backtests can score all fixtures without silently selecting away hard games.
    result["abstain"] = result["low_confidence"]
    result["prediction_set"] = np.where(result["low_confidence"], "LOW_CONFIDENCE", "STANDARD")
    result["prediction_time_utc"] = now.isoformat()
    result["matchday_status"] = matchday_diagnostics["status"].astype(str).to_numpy()
    result["matchday_applied"] = matchday_diagnostics["applied"].astype(bool).to_numpy()
    result["matchday_freshness"] = pd.to_numeric(matchday_diagnostics["freshness"], errors="coerce").to_numpy()
    result["matchday_signal_count"] = pd.to_numeric(matchday_diagnostics["signal_count"], errors="coerce").fillna(0).astype(int).to_numpy()
    result["matchday_adjustment_l1"] = pd.to_numeric(matchday_diagnostics["adjustment_l1"], errors="coerce").fillna(0.0).to_numpy()
    result["matchday_source"] = matchday_diagnostics["source"].astype(str).to_numpy()
    result["matchday_signals"] = matchday_diagnostics["signal_names"].astype(str).to_numpy()
    # Legacy schema-1 bundles remain valid for historical 1X2 tests only. Production secondary outputs require schema 2.
    if bundle.get("schema_version", 1) >= 2:
        if "score_model" not in bundle:
            raise RuntimeError("Production bundle lacks the locked Score model")
        # MOM is a secondary layer. Missing upstream PIT-safe player evidence
        # must not suppress validated 1X2/Score/O-U/BTTS outputs.
        # The per-fixture loop below marks MOM as BLOCKED_UPSTREAM_PLAYER_MODEL.
    else:
        result["model_version"] = str(bundle["model_version"])
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        temp_output = output_file.with_suffix(output_file.suffix + ".tmp")
        result.to_csv(temp_output, index=False)
        temp_output.replace(output_file)
        return _write_status(status_file, "PREDICTED", prediction_time_utc=now.isoformat(), source_rows=int(len(fixtures)), eligible_rows=int(len(eligible)), prediction_rows=int(len(result)), standard_rows=int((~result["low_confidence"]).sum()), low_confidence_rows=int(result["low_confidence"].sum()), abstained_rows=int(result["abstain"].sum()), output_path=str(output_file), model_version=str(bundle["model_version"]), oos_claimed=bool(registry.get("oos_verified", False)), model_mode=model_mode, matchday_merge_status=matchday_merge_status)
    selected_score_method = str(bundle.get("score_method", "primary"))
    if selected_score_method == "neutral_aware":
        if "neutral_venue" not in eligible.columns:
            raise RuntimeError("Neutral-aware Score production requires explicit neutral_venue fixture evidence")
        neutral_missing = eligible["neutral_venue"].isna()
        if bool(neutral_missing.any()):
            raise RuntimeError("Neutral-aware Score production blocked because neutral_venue is unknown for an eligible fixture")

    score_rows = []
    score_market_rows = []
    mom_rows = []
    mom_statuses = []
    mom_input_available = "mom_candidates_json" in eligible.columns
    for row in eligible.itertuples(index=False):
        neutral_venue = getattr(row, "neutral_venue", False) if hasattr(row, "neutral_venue") else False
        if pd.isna(neutral_venue):
            neutral_venue = False
        score_rows.append(
            predict_score_candidates(
                bundle["score_model"], row.home_team, row.away_team, row.competition,
                neutral_venue=bool(neutral_venue),
            )
        )
        score_market_rows.append(
            predict_score_markets(
                bundle["score_model"], row.home_team, row.away_team, row.competition,
                neutral_venue=bool(neutral_venue),
            )
        )
        if not mom_input_available:
            # MOM is an optional secondary layer. Missing upstream player probabilities
            # must block only MOM, never the validated 1X2/Score/O-U/BTTS outputs.
            mom_rows.append([])
            mom_statuses.append("BLOCKED_UPSTREAM_PLAYER_MODEL")
            continue
        raw_mom = getattr(row, "mom_candidates_json", None)
        if raw_mom is None or (isinstance(raw_mom, float) and np.isnan(raw_mom)) or str(raw_mom).strip() == "":
            mom_rows.append([])
            mom_statuses.append("BLOCKED_UPSTREAM_PLAYER_MODEL")
            continue
        try:
            selected_mom = predict_mom_candidates(raw_mom)
            if not selected_mom:
                mom_rows.append([])
                mom_statuses.append("BLOCKED_UPSTREAM_PLAYER_MODEL")
            else:
                mom_rows.append(selected_mom)
                mom_statuses.append("PREDICTED")
        except Exception:
            # Fail closed at the MOM layer: invalid player evidence cannot be converted
            # into guessed players/probabilities and cannot suppress other outputs.
            mom_rows.append([])
            mom_statuses.append("BLOCKED_INVALID_UPSTREAM_EVIDENCE")
    for rank in range(1, 4):
        result[f"score_{rank}"] = [f"{x[rank-1]['home_goals']}-{x[rank-1]['away_goals']}" for x in score_rows]
        result[f"score_{rank}_probability"] = [x[rank-1]["probability"] for x in score_rows]
    for key in ("over_0_5", "under_0_5", "over_1_5", "under_1_5", "over_2_5", "under_2_5", "over_3_5", "under_3_5", "over_4_5", "under_4_5", "btts_yes", "btts_no"):
        result[f"market_{key}"] = [float(x[key]) for x in score_market_rows]
    for rank in range(1, 5):
        result[f"mom_{rank}_player_id"] = [
            x[rank - 1]["player_id"] if len(x) >= rank else pd.NA
            for x in mom_rows
        ]
        result[f"mom_{rank}_probability"] = [
            float(x[rank - 1]["probability"]) if len(x) >= rank else np.nan
            for x in mom_rows
        ]
    result["mom_status"] = mom_statuses
    result["score_top3_probability_mass"] = sum(result[f"score_{rank}_probability"] for rank in range(1, 4))
    result["mom_top4_probability_mass"] = sum(result[f"mom_{rank}_probability"] for rank in range(1, 5))
    result["model_version"] = str(bundle["model_version"])
    if len(result) != len(eligible) or result["match_id"].duplicated().any():
        raise RuntimeError("Production prediction output failed fixture identity invariants")
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
        standard_rows=int((~result["low_confidence"]).sum()),
        low_confidence_rows=int(result["low_confidence"].sum()),
        abstained_rows=int(result["abstain"].sum()),
        output_path=str(output_file),
        model_version=str(bundle["model_version"]),
        model_mode=model_mode,
        oos_claimed=bool(registry.get("oos_verified", False)),
        matchday_merge_status=matchday_merge_status,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", default="artifacts/future_fixtures.csv")
    parser.add_argument("--bundle", default="artifacts/production_model.pkl")
    parser.add_argument("--output", default="artifacts/predictions.csv")
    parser.add_argument("--status", default="artifacts/prediction_status.json")
    parser.add_argument("--prediction-time", default=None)
    parser.add_argument("--registry", default="artifacts/model_registry.json")
    parser.add_argument("--model-policy", choices=["production", "best_available"], default="best_available")
    args = parser.parse_args()
    result = run(args.fixtures, args.bundle, args.output, args.status, args.prediction_time, args.registry, args.model_policy)
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
