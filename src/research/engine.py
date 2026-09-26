from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.coverage import build_coverage
from src.data.football_data import load_available_history
from src.data.fixture_field_audit import run_audit
from src.data.pit_source_adapter import competition_adapter_matrix
from src.data.source_registry import SOCCER_SOURCES
from src.evaluation.score_walk_forward import run_score_walk_forward
from src.evaluation.walk_forward import TARGET_ACCURACY, run_walk_forward
from src.features.soccer_features import add_target, build_match_features
from src.prediction.model_bundle import train_and_save_bundle
from src.research.adoption import adoption_decision
from src.research.llm import weakness_advice
from src.research.registry import save_registry
from src.research.score_model_selection import select_score_model, verify_selected_score_model
from src.research.stability_gate import evaluate_stability

EXCLUDED_MODEL_COLUMNS = {"match_id", "competition", "season", "season_start", "kickoff_utc", "home_team", "away_team", "prediction_cutoff_at_utc", "home_goals", "away_goals", "target", "pit_verified", "feature_source_max_available_at_utc"}
PIT_POLICY = "explicit_source_publication_time_only; unknown_publication_time_excluded"

PRIMARY_SCORE_METRICS = (
    "score_logloss",
    "exact_score_hit_rate",
    "top3_score_hit_rate",
    "top4_score_hit_rate",
    "home_goals_mae",
    "away_goals_mae",
    "total_goals_mae",
    "over_2_5_logloss",
    "over_2_5_brier",
    "btts_logloss",
    "btts_brier",
)


def _primary_score_metrics_finite(score_oos: pd.DataFrame, *, min_blocks: int = 1) -> bool:
    if score_oos.empty or len(score_oos) < int(min_blocks):
        return False
    missing = [column for column in PRIMARY_SCORE_METRICS if column not in score_oos.columns]
    if missing:
        return False
    values = score_oos[list(PRIMARY_SCORE_METRICS)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return False
    return True


def _oos_temporal_integrity(frame: pd.DataFrame, *, locked_blocks: int = 2) -> dict:
    """Verify chronological OOS block ordering and development/locked separation.

    This is deliberately fail-closed: missing or unparsable block boundaries are
    a failure, because chronological ordering cannot be inferred from row order.
    """
    result = {
        "status": "FAIL",
        "fail_closed": True,
        "blocks": int(len(frame)) if frame is not None else 0,
        "locked_blocks": int(locked_blocks),
        "has_boundaries": False,
        "all_intervals_valid": False,
        "chronological": False,
        "non_overlapping": False,
        "development_before_locked": False,
        "reason": "",
    }
    if frame is None or frame.empty:
        result["reason"] = "oos_frame_empty"
        return result
    required = {"oos_start", "oos_end"}
    if not required.issubset(frame.columns):
        result["reason"] = "missing_oos_boundaries"
        return result
    starts = pd.to_datetime(frame["oos_start"], utc=True, errors="coerce")
    ends = pd.to_datetime(frame["oos_end"], utc=True, errors="coerce")
    result["has_boundaries"] = bool(starts.notna().all() and ends.notna().all())
    if not result["has_boundaries"]:
        result["reason"] = "unparseable_oos_boundaries"
        return result

    intervals_valid = bool((starts <= ends).all())
    result["all_intervals_valid"] = intervals_valid
    if not intervals_valid:
        result["reason"] = "oos_start_after_oos_end"
        return result

    start_ns = starts.astype("int64").to_numpy()
    end_ns = ends.astype("int64").to_numpy()
    chronological = bool(np.all(start_ns[1:] > start_ns[:-1])) if len(starts) > 1 else True
    result["chronological"] = chronological
    if not chronological:
        result["reason"] = "oos_blocks_not_strictly_chronological"
        return result

    non_overlapping = bool(np.all(start_ns[1:] > end_ns[:-1])) if len(starts) > 1 else True
    result["non_overlapping"] = non_overlapping
    if not non_overlapping:
        result["reason"] = "oos_blocks_overlap"
        return result

    if len(frame) < int(locked_blocks) + 1:
        result["reason"] = "insufficient_blocks_for_development_and_locked_holdout"
        return result

    development_end = ends.iloc[: -int(locked_blocks)].max()
    locked_start = starts.iloc[-int(locked_blocks):].min()
    development_before_locked = bool(development_end < locked_start)
    result["development_before_locked"] = development_before_locked
    if not development_before_locked:
        result["reason"] = "development_overlaps_locked_holdout"
        return result

    result["status"] = "PASS"
    result["reason"] = "strict_chronological_non_overlapping_oos"
    return result


def snapshot_id(df: pd.DataFrame) -> str:
    excluded = {"retrieved_at_utc", "source_available_at_utc", "pit_evidence_url", "capture_digest"}
    stable = df[[c for c in df.columns if c not in excluded]].copy()
    for c in stable.columns:
        if pd.api.types.is_datetime64_any_dtype(stable[c]):
            stable[c] = pd.to_datetime(stable[c], utc=True, errors="coerce").astype("string")
    sort_cols = [c for c in ["competition", "season_start", "kickoff_utc", "home_team", "away_team", "match_id"] if c in stable.columns]
    if sort_cols:
        stable = stable.sort_values(sort_cols, kind="mergesort")
    return hashlib.sha256(stable.to_json(orient="records", date_format="iso").encode()).hexdigest()


def _write_source_registry(out: Path) -> None:
    pd.DataFrame([{"name": s.name, "kind": s.kind, "role": s.role, "fields": ",".join(s.fields), "historical": s.historical, "pit_capable": s.pit_capable, "live_capable": s.live_capable, "auth_required": s.auth_required, "primary_for": ",".join(s.primary_for), "notes": s.notes} for s in SOCCER_SOURCES]).to_csv(out / "source_registry.csv", index=False)


def _model_features(feats: pd.DataFrame) -> list[str]:
    numeric = feats.select_dtypes(include=["number", "bool"]).columns.tolist()
    cols = [c for c in numeric if c not in EXCLUDED_MODEL_COLUMNS and not c.startswith("baseline_")]
    if not cols:
        raise ValueError("No numeric model features available")
    return cols


def _load_preflight_pit_features(out: Path, history: pd.DataFrame) -> pd.DataFrame | None:
    """Load the PIT-replayed feature matrix produced by completion_gate.

    The completion gate enriches historical rows with explicit publication
    evidence before building pit_replay_features.csv. Re-loading raw history
    here would discard that evidence and can incorrectly reduce pit_verified to
    zero. The handoff is validated by exact match identity and outcome
    availability; no inferred timestamps are introduced.
    """
    path = out / "pit_replay_features.csv"
    if not path.is_file() or path.stat().st_size <= 0:
        return None

    features = pd.read_csv(path)
    required_features = {
        "match_id",
        "pit_verified",
        "prediction_cutoff_at_utc",
        "kickoff_utc",
        "feature_source_max_available_at_utc",
    }
    missing = sorted(required_features - set(features.columns))
    if missing:
        raise RuntimeError(
            f"PIT preflight feature handoff is missing required columns: {missing}"
        )
    if features["match_id"].isna().any() or features["match_id"].astype(str).str.strip().eq("").any():
        raise RuntimeError("PIT preflight feature handoff contains missing/empty match_id values")
    if features["match_id"].duplicated().any():
        raise RuntimeError("PIT preflight feature handoff contains duplicate match_id values")

    outcomes = history[["match_id", "home_goals", "away_goals"]].copy()
    if outcomes["match_id"].isna().any() or outcomes["match_id"].duplicated().any():
        raise RuntimeError("Historical outcome table has missing/duplicate match_id values")

    # The target match's own publication time is not part of the prediction
    # feature PIT condition: its outcome is the label being scored. The actual
    # PIT condition is that every historical feature consumed for this target was
    # available by the prediction cutoff. The preflight feature handoff exposes
    # that invariant as feature_source_max_available_at_utc.
    verified = features["pit_verified"].eq(True)
    cutoff = pd.to_datetime(features["prediction_cutoff_at_utc"], utc=True, errors="coerce")
    feature_available = pd.to_datetime(
        features["feature_source_max_available_at_utc"], utc=True, errors="coerce"
    )
    invalid_feature_pit = verified & (
        cutoff.isna()
        | feature_available.isna()
        | (feature_available > cutoff)
    )
    if invalid_feature_pit.any():
        raise RuntimeError(
            "PIT preflight feature handoff contains invalid feature_source_max_available_at_utc "
            f"for {int(invalid_feature_pit.sum())} PIT-verified rows"
        )

    # Keep source publication evidence when it is present in the historical table,
    # but do not require it for the target label row: source availability of the
    # outcome itself is not the predictor-side PIT condition.
    if "source_available_at_utc" in history.columns:
        pit_timing = history[["match_id", "source_available_at_utc"]].copy()
        pit_timing["match_id"] = pit_timing["match_id"].astype(str)
        pit_timing["source_available_at_utc"] = pd.to_datetime(
            pit_timing["source_available_at_utc"], utc=True, errors="coerce"
        )
    else:
        pit_timing = pd.DataFrame({
            "match_id": history["match_id"].astype(str),
            "source_available_at_utc": pd.NaT,
        })

    # build_match_features focuses on model features and may omit publication
    # evidence metadata. Restore optional source publication evidence from the
    # historical table without weakening predictor-side PIT validation.
    features = features.drop(columns=["source_available_at_utc"], errors="ignore")
    merged = features.merge(
        outcomes,
        on="match_id",
        how="left",
        validate="one_to_one",
    ).merge(
        pit_timing,
        on="match_id",
        how="left",
        validate="one_to_one",
    )
    if len(merged) != len(features):
        raise RuntimeError("PIT preflight feature handoff changed row count during outcome join")
    if merged[["home_goals", "away_goals"]].isna().any().any():
        missing_outcomes = int(merged[["home_goals", "away_goals"]].isna().any(axis=1).sum())
        raise RuntimeError(
            f"PIT preflight feature handoff has {missing_outcomes} rows without historical outcomes"
        )

    home_goals = pd.to_numeric(merged["home_goals"], errors="coerce")
    away_goals = pd.to_numeric(merged["away_goals"], errors="coerce")
    merged["target"] = np.select(
        [
            home_goals.isna() | away_goals.isna(),
            home_goals > away_goals,
            home_goals == away_goals,
        ],
        [np.nan, 0, 1],
        default=2,
    )
    merged["pit_verified"] = merged["pit_verified"].astype(bool)
    return merged

def _archive_audit_sample(history: pd.DataFrame, out: Path) -> dict:
    if os.getenv("PIT_ENABLE_ARCHIVE_AUDIT", "0") != "1":
        return {"status": "SKIPPED", "reason": "disabled_in_main_research"}
    try:
        rows_per_group = max(1, int(os.getenv("PIT_REPLAY_ROWS_PER_GROUP", "1")))
    except ValueError:
        rows_per_group = 1
    sample = history.sort_values(["competition", "season_start", "kickoff_utc"], kind="mergesort").groupby(["competition", "season_start"], sort=False, dropna=False, group_keys=False).head(rows_per_group).copy()
    try:
        from src.data.pit_source_adapter import apply_pit_evidence, build_pit_diagnostic
        replayed = apply_pit_evidence(sample)
        build_pit_diagnostic(sample).to_csv(out / "pit_diagnostic_sample.csv", index=False)
        if os.getenv("PIT_ENABLE_SECONDARY_ARCHIVE", "0") == "1":
            from src.data.pit_archive_fallback import apply_arquivo_fallback
            replayed = apply_arquivo_fallback(replayed)
        replayed.to_csv(out / "normalized_history_archive_audit.csv", index=False)
        return {"status": "COMPLETED", "rows": int(len(replayed))}
    except Exception as exc:
        return {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}


def _write_status(out: Path, report: dict) -> None:
    (out / "run_status.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _build_research_gates(
    out: Path,
    *,
    history_nonempty: bool,
    pit_verified: int,
    wf: pd.DataFrame,
    development_oos: pd.DataFrame,
    locked_oos: pd.DataFrame,
    selections: pd.DataFrame,
    stability: dict,
    adoption: dict,
    model_bundle: dict | None,
    audit_report: dict,
) -> dict[str, bool]:
    """Build the exact gate map consumed by the fail-closed production contract."""
    completion = {}
    audit_gate = {}
    try:
        completion = json.loads((out / "completion_gate.json").read_text(encoding="utf-8"))
    except Exception:
        completion = {}
    try:
        audit_gate = json.loads((out / "audit_gate.json").read_text(encoding="utf-8"))
    except Exception:
        audit_gate = {}

    required_artifacts = (
        "oos_metrics.csv",
        "model_selection.csv",
        "development_oos_metrics.csv",
        "locked_oos_metrics.csv",
        "score_oos_metrics.csv",
        "score_oos_gate.json",
        "score_model_selection.json",
        "score_locked_gate.json",
        "candidate_lock.json",
        "adoption_decision.json",
        "oos_temporal_integrity.json",
        "score_oos_temporal_integrity.json",
        "calibration_gate.json",
    )
    artifacts_present = all((out / name).is_file() and (out / name).stat().st_size > 0 for name in required_artifacts)

    data_ok = bool(history_nonempty and pit_verified > 0)
    schema_ok = bool(audit_report.get("audit_execution_ok") is True and audit_gate.get("full_gate_passed") is True)
    leakage_ok = bool(
        completion.get("full_gate_passed") is True
        and completion.get("pit_publication_time_gate") is True
        and pit_verified > 0
    )
    features_ok = bool((out / "pit_replay_features.csv").is_file() and (out / "pit_replay_features.csv").stat().st_size > 0)
    training_ok = bool(not selections.empty and all(isinstance(x, dict) for x in selections.to_dict(orient="records")))
    backtest_ok = bool(not wf.empty)
    score_oos_gate = json.loads((out / "score_oos_gate.json").read_text(encoding="utf-8")) if (out / "score_oos_gate.json").exists() else {}
    score_locked_gate = json.loads((out / "score_locked_gate.json").read_text(encoding="utf-8")) if (out / "score_locked_gate.json").exists() else {}
    oos_temporal = json.loads((out / "oos_temporal_integrity.json").read_text(encoding="utf-8")) if (out / "oos_temporal_integrity.json").exists() else {}
    score_temporal = json.loads((out / "score_oos_temporal_integrity.json").read_text(encoding="utf-8")) if (out / "score_oos_temporal_integrity.json").exists() else {}
    oos_ok = bool(
        len(wf) >= 3
        and len(development_oos) >= 1
        and len(locked_oos) == 2
        and score_oos_gate.get("status") == "PASS"
        and int(score_oos_gate.get("blocks", 0)) >= 5
        and score_oos_gate.get("block_rows_ok") is True
        and score_oos_gate.get("finite_metrics") is True
        and score_locked_gate.get("status") == "PASS"
        and oos_temporal.get("status") == "PASS"
        and score_temporal.get("status") == "PASS"
    )
    prediction_ok = bool(
        str(adoption.get("status", "")).upper() == "ADOPT"
        and isinstance(model_bundle, dict)
        and model_bundle.get("status") != "ERROR"
        and (out / "production_model.pkl").is_file()
        and (out / "production_model.pkl").stat().st_size > 0
    )
    sanity_ok = bool(
        stability.get("status") == "PASS"
        and np.isfinite(pd.to_numeric(wf.get("logloss"), errors="coerce")).all()
        and np.isfinite(pd.to_numeric(wf.get("brier"), errors="coerce")).all()
    )

    return {
        "data": data_ok,
        "schema": schema_ok,
        "leakage": leakage_ok,
        "features": features_ok,
        "training": training_ok,
        "backtest": backtest_ok,
        "oos": oos_ok,
        "prediction": prediction_ok,
        "sanity": sanity_ok,
        "artifact": artifacts_present,
    }


def run(out_dir: str = "artifacts") -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); _write_source_registry(out)
    try:
        audit_report = run_audit(str(out))
    except Exception as exc:
        audit_report = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}", "audit_complete": False}
    (out / "engine_audit_summary.json").write_text(json.dumps(audit_report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    competition_adapter_matrix().to_csv(out / "pit_competition_adapter_matrix.csv", index=False)
    history, acquisition = load_available_history(); acquisition.to_csv(out / "acquisition_coverage.csv", index=False)
    coverage = build_coverage(history); coverage.to_csv(out / "coverage_matrix.csv", index=False)
    if history.empty:
        report = {"status": "BLOCKED", "reason": "No historical data acquired", "oos_claimed": False, "audit": audit_report}; _write_status(out, report); return report
    archive_audit = _archive_audit_sample(history, out)
    preflight_feats = _load_preflight_pit_features(out, history)
    if preflight_feats is not None:
        feats = preflight_feats
    else:
        # Backward-compatible fallback for direct/local engine invocation.
        # The production workflow normally reaches this branch only when the
        # preflight artifact is unavailable, in which case PIT remains fail-closed.
        feats = add_target(build_match_features(history, history), history)
        feats.to_csv(out / "pit_replay_features.csv", index=False)
    pit_verified = int(feats["pit_verified"].sum()) if "pit_verified" in feats.columns else 0; pit_total = int(len(feats))
    if pit_verified == 0:
        report = {"status": "BLOCKED", "reason": "No match rows have sufficient historical result state under deterministic PIT.", "acquired_rows": int(len(history)), "snapshot_id": snapshot_id(history), "pit_policy": PIT_POLICY, "pit_verified_rows": 0, "pit_verified_rate": 0.0, "archive_audit": archive_audit, "audit": audit_report, "oos_claimed": False}; report["ai_research"] = weakness_advice(report); _write_status(out, report); return report
    minimum_score_blocks = 5
    minimum_score_rows_per_block = 500
    score_oos = pd.DataFrame()
    primary_finite = False
    score_block_rows = pd.Series(dtype=float)
    score_block_rows_ok = False
    try:
        score_oos = run_score_walk_forward(
            feats,
            min_train=max(500, int(os.getenv("SOCCER_SCORE_MIN_TRAIN", "1000"))),
            oos_block=max(500, int(os.getenv("SOCCER_SCORE_OOS_BLOCK", "2000"))),
        )
        score_oos.to_csv(out / "score_oos_metrics.csv", index=False)
        primary_finite = _primary_score_metrics_finite(score_oos)
        enough_score_blocks = len(score_oos) >= minimum_score_blocks
        score_block_rows = pd.to_numeric(score_oos["n"], errors="coerce") if "n" in score_oos.columns else pd.Series(dtype=float)
        score_block_rows_ok = bool(
            len(score_block_rows) > 0
            and score_block_rows.notna().all()
            and (score_block_rows >= minimum_score_rows_per_block).all()
        )
        score_oos_status = {
            "status": "PASS" if primary_finite and enough_score_blocks and score_block_rows_ok else "ERROR",
            "blocks": int(len(score_oos)),
            "minimum_total_blocks": minimum_score_blocks,
            "minimum_rows_per_block": minimum_score_rows_per_block,
            "block_rows": [int(x) if pd.notna(x) else None for x in score_block_rows.tolist()],
            "block_rows_ok": score_block_rows_ok,
            "development_blocks_expected": 3,
            "locked_blocks": 2,
            "rows": int(score_oos["n"].sum()) if "n" in score_oos.columns else 0,
            "finite_metrics": primary_finite,
            "primary_metrics_finite": primary_finite,
            "recency_status": sorted(set(score_oos["recency_status"].astype(str))) if "recency_status" in score_oos.columns else [],
            "dc_status": sorted(set(score_oos["dc_status"].astype(str))) if "dc_status" in score_oos.columns else [],
        }
    except Exception as exc:
        score_oos = pd.DataFrame()
        primary_finite = False
        score_block_rows = pd.Series(dtype=float)
        score_block_rows_ok = False
        score_oos_status = {
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "blocks": 0,
            "minimum_total_blocks": minimum_score_blocks,
            "minimum_rows_per_block": minimum_score_rows_per_block,
            "block_rows": [],
            "block_rows_ok": False,
            "rows": 0,
            "finite_metrics": False,
            "primary_metrics_finite": False,
        }
    (out / "score_oos_gate.json").write_text(
        json.dumps(score_oos_status, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    score_protocol_ready = len(score_oos) >= 5 and primary_finite and score_block_rows_ok
    score_development_oos = score_oos.iloc[:-2].copy() if score_protocol_ready else pd.DataFrame()
    score_selection = select_score_model(score_development_oos, min_rows_per_block=minimum_score_rows_per_block)
    score_locked_oos = score_oos.tail(2).copy() if score_protocol_ready else pd.DataFrame()
    score_selection["protocol"] = {
        "minimum_total_blocks": 5,
        "development_blocks": int(len(score_development_oos)),
        "locked_blocks": int(len(score_locked_oos)),
        "locked_oos_untouched_for_selection": True,
    }
    score_locked_gate = verify_selected_score_model(score_selection, score_locked_oos, min_rows_per_block=minimum_score_rows_per_block)
    score_temporal = _oos_temporal_integrity(score_oos, locked_blocks=2)
    (out / "score_oos_temporal_integrity.json").write_text(
        json.dumps(score_temporal, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (out / "score_model_selection.json").write_text(
        json.dumps(score_selection, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (out / "score_locked_gate.json").write_text(
        json.dumps(score_locked_gate, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    case_output_path = out / "oos_case_diagnostics.csv"
    wf, selections = run_walk_forward(feats, _model_features(feats), case_output_path=str(case_output_path))
    oos_temporal = _oos_temporal_integrity(wf, locked_blocks=2)
    (out / "oos_temporal_integrity.json").write_text(
        json.dumps(oos_temporal, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    wf.to_csv(out / "oos_metrics.csv", index=False)
    selections.to_csv(out / "model_selection.csv", index=False)
    if len(wf) < 3 or oos_temporal.get("status") != "PASS":
        report = {"status": "BLOCKED", "reason": "At least three chronological OOS blocks are required: development plus two locked holdout blocks.", "snapshot_id": snapshot_id(history), "pit_policy": PIT_POLICY, "pit_verified_rows": pit_verified, "pit_total_rows": pit_total, "archive_audit": archive_audit, "audit": audit_report, "oos_claimed": False}; _write_status(out, report); return report
    development_oos = wf.iloc[:-2].copy(); locked = wf.tail(2).copy(); development_oos.to_csv(out / "development_oos_metrics.csv", index=False); locked.to_csv(out / "locked_oos_metrics.csv", index=False)
    # Explicit calibration evidence for production readiness. Calibration is learned
    # only from the disjoint validation calibration slice and never from locked OOS.
    latest_selection = selections.iloc[-1].to_dict() if not selections.empty else {}
    try:
        calibration_temperature = float(latest_selection.get("temperature", float("nan")))
    except (TypeError, ValueError):
        calibration_temperature = float("nan")
    try:
        calibration_rows = int(latest_selection.get("calibration_rows", 0))
    except (TypeError, ValueError):
        calibration_rows = 0
    calibration_used = bool(latest_selection.get("temperature_calibration_used", False))
    contextual_temperatures = latest_selection.get("contextual_temperatures") or {}
    contextual_temps_valid = False
    if isinstance(contextual_temperatures, dict):
        try:
            contextual_temps_valid = all(
                np.isfinite(float(value)) and 0.70 <= float(value) <= 1.60
                for value in contextual_temperatures.values()
            )
        except (TypeError, ValueError):
            contextual_temps_valid = False
    calibration_gate = {
        "status": "PASS" if (
            np.isfinite(calibration_temperature)
            and 0.70 <= calibration_temperature <= 1.60
            and calibration_rows >= 60
            and isinstance(latest_selection.get("temperature"), (int, float, np.number))
            and contextual_temps_valid
        ) else "FAIL",
        "method": "temperature_scaling_on_disjoint_validation_calibration_slice",
        "temperature": calibration_temperature,
        "calibration_rows": calibration_rows,
        "temperature_calibration_used": calibration_used,
        "contextual_temperatures": contextual_temperatures,
        "contextual_temperatures_valid": contextual_temps_valid,
        "source": "model_selection.csv:last_validation_fold_selection_record",
        "locked_oos_used_for_calibration": False,
        "fail_closed": True,
    }
    (out / "calibration_gate.json").write_text(
        json.dumps(calibration_gate, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    stability_folds = []
    for _, row in wf.iterrows():
        stability_folds.append({
            "league": row.get("leagues", ""),
            "season": row.get("seasons", ""),
            "baseline": {
                "logloss": row.get("baseline_logistic_logloss"),
                "brier": row.get("baseline_logistic_brier"),
                "accuracy": row.get("baseline_logistic_accuracy"),
            },
            "candidate": {
                "logloss": row.get("logloss"),
                "brier": row.get("brier"),
                "accuracy": row.get("accuracy"),
            },
        })
    stability = evaluate_stability(stability_folds)
    (out / "stability_gate.json").write_text(json.dumps(stability, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    candidate_lock = {"status": "LOCKED", "selection_source": "historical_validation_only", "selection_artifact": "model_selection.csv", "development_oos_blocks": int(len(development_oos)), "locked_oos_blocks": int(len(locked)), "locked_oos_untouched": True, "target_accuracy": TARGET_ACCURACY, "model_family": "validation-selected calibrated ensemble", "feature_policy": "PIT-safe numeric features only", "pit_policy": PIT_POLICY}
    (out / "candidate_lock.json").write_text(json.dumps(candidate_lock, indent=2, ensure_ascii=False), encoding="utf-8")
    baseline_cols = ["oos_start", "oos_end", "baseline_logistic_logloss", "baseline_logistic_accuracy", "baseline_logistic_brier", "baseline_logistic_rps", "baseline_logistic_ece", "n"]
    candidate_cols = ["oos_start", "oos_end", "logloss", "accuracy", "brier", "rps", "ece", "n"]
    baseline = locked[baseline_cols].rename(columns={"baseline_logistic_logloss": "logloss", "baseline_logistic_accuracy": "accuracy", "baseline_logistic_brier": "brier", "baseline_logistic_rps": "rps", "baseline_logistic_ece": "ece"})
    candidate = locked[candidate_cols].copy()
    adoption = adoption_decision(baseline, candidate, development_oos=development_oos, min_accuracy=TARGET_ACCURACY, min_locked_rows_per_block=500)
    adoption["external_stability_gate"] = stability
    (out / "adoption_decision.json").write_text(json.dumps(adoption, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    model_bundle = None
    validated_candidate_bundle = None
    if not selections.empty and len(locked) == 2 and oos_temporal.get("status") == "PASS" and calibration_gate.get("status") == "PASS":
        # Reference-prediction candidate: train only on data strictly before the
        # locked OOS horizon. Locked outcomes remain untouched, even when the
        # candidate later fails the adoption comparison.
        selection = selections.iloc[-1].to_dict()
        locked_start = pd.to_datetime(locked["oos_start"], utc=True, errors="coerce").min()
        candidate_training = feats.copy()
        candidate_training["kickoff_utc"] = pd.to_datetime(candidate_training["kickoff_utc"], utc=True, errors="coerce")
        candidate_training = candidate_training[
            (candidate_training["kickoff_utc"] < locked_start) & (candidate_training["pit_verified"] == True)
        ].copy()
        candidate_version = hashlib.sha256(
            json.dumps(
                {
                    "snapshot_id": snapshot_id(history),
                    "selection": selection,
                    "score_selection": score_selection,
                    "score_locked_gate": score_locked_gate,
                    "training_cutoff": str(locked_start),
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()[:16]
        try:
            validated_candidate_bundle = train_and_save_bundle(
                candidate_training,
                _model_features(candidate_training),
                selection,
                str(out / "validated_candidate_model.pkl"),
                candidate_version,
                snapshot_id(history),
                score_selection=score_selection,
                score_locked_gate=score_locked_gate,
            )
            (out / "validated_candidate_model.json").write_text(
                json.dumps({**validated_candidate_bundle, "status": "VALIDATED_CANDIDATE"}, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            candidate_registry = save_registry(
                str(out / "validated_candidate_registry.json"),
                model_version=candidate_version,
                feature_version="pit_safe_v1",
                research_cycle=str(pd.Timestamp.utcnow().isoformat()),
                git_commit_sha=os.getenv("GITHUB_SHA", "unknown"),
                data_snapshot_id=snapshot_id(history),
                metrics={"development_oos": development_oos.to_dict(orient="records"), "locked_oos_verification": locked.to_dict(orient="records")},
                adoption_status="VALIDATED_CANDIDATE",
                parameters={
                    "weights": validated_candidate_bundle.get("weights", {}),
                    "feature_count": validated_candidate_bundle.get("feature_count"),
                    "fit_rows": validated_candidate_bundle.get("fit_rows"),
                    "score_method": validated_candidate_bundle.get("score_method", "primary"),
                    "routing_policy": validated_candidate_bundle.get("routing_policy"),
                },
                training_end=validated_candidate_bundle.get("fit_end"),
                calibration={
                    "temperature": validated_candidate_bundle.get("temperature"),
                    "score_method": validated_candidate_bundle.get("score_method", "primary"),
                    "gate": calibration_gate,
                },
            )
            (out / "validated_candidate_registry.json").write_text(
                json.dumps({**candidate_registry, "adoption_status": "VALIDATED_CANDIDATE"}, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            candidate_provenance = {
                "provenance_schema_version": 1,
                "status": "VALIDATED_CANDIDATE",
                "model_version": candidate_version,
                "data_snapshot_id": snapshot_id(history),
                "training_cutoff_exclusive_utc": str(locked_start),
                "locked_oos_used_for_training": False,
                "calibration_gate": calibration_gate,
                "files": {
                    "validated_candidate_model.pkl": {"sha256": _sha256(out / "validated_candidate_model.pkl"), "bytes": (out / "validated_candidate_model.pkl").stat().st_size},
                    "validated_candidate_model.json": {"sha256": _sha256(out / "validated_candidate_model.json"), "bytes": (out / "validated_candidate_model.json").stat().st_size},
                    "validated_candidate_registry.json": {"sha256": _sha256(out / "validated_candidate_registry.json"), "bytes": (out / "validated_candidate_registry.json").stat().st_size},
                },
            }
            (out / "validated_candidate_provenance.json").write_text(
                json.dumps(candidate_provenance, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
            )
        except Exception as exc:
            validated_candidate_bundle = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
    if adoption.get("status") == "ADOPT" and not selections.empty:
        selection = selections.iloc[-1].to_dict()
        model_version = hashlib.sha256(
            json.dumps(
                {
                    "snapshot_id": snapshot_id(history),
                    "selection": selection,
                    "score_selection": score_selection,
                    "score_locked_gate": score_locked_gate,
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()[:16]
        try:
            model_bundle = train_and_save_bundle(
                feats,
                _model_features(feats),
                selection,
                str(out / "production_model.pkl"),
                model_version,
                snapshot_id(history),
                score_selection=score_selection,
                score_locked_gate=score_locked_gate,
            )
            registry = save_registry(
                str(out / "model_registry.json"),
                model_version=model_version,
                feature_version="pit_safe_v1",
                research_cycle=str(pd.Timestamp.utcnow().isoformat()),
                git_commit_sha=os.getenv("GITHUB_SHA", "unknown"),
                data_snapshot_id=snapshot_id(history),
                metrics={"locked_oos": locked.to_dict(orient="records"), "development_oos": development_oos.to_dict(orient="records")},
                adoption_status="ADOPT",
                parameters={
                    "weights": model_bundle.get("weights", {}),
                    "feature_count": model_bundle.get("feature_count"),
                    "fit_rows": model_bundle.get("fit_rows"),
                    "score_method": model_bundle.get("score_method", "primary"),
                    "routing_policy": model_bundle.get("routing_policy"),
                },
                training_end=model_bundle.get("fit_end"),
                feature_cols=_model_features(feats),
                calibration={
                    "temperature": model_bundle.get("temperature"),
                    "score_method": model_bundle.get("score_method", "primary"),
                },
            )
            (out / "production_model.json").write_text(json.dumps({**model_bundle, "adoption_status": "ADOPT", "registry": registry}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        except Exception as exc:
            model_bundle = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
            (out / "production_model.json").write_text(json.dumps(model_bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    oos_n = int(wf["n"].sum()) if "n" in wf.columns else 0; oos_acc = float(np.average(wf["accuracy"], weights=wf["n"])) if oos_n else float("nan")
    locked_n = int(locked["n"].sum()); locked_acc = float(np.average(locked["accuracy"], weights=locked["n"])) if locked_n else float("nan")
    gates = _build_research_gates(
        out,
        history_nonempty=not history.empty,
        pit_verified=pit_verified,
        wf=wf,
        development_oos=development_oos,
        locked_oos=locked,
        selections=selections,
        stability=stability,
        adoption=adoption,
        model_bundle=model_bundle,
        audit_report=audit_report,
    )
    oos_claimed = bool(gates["oos"])
    report = {
        "status": "OK" if oos_claimed else "DEGRADED",
        "snapshot_id": snapshot_id(history),
        "oos": wf.mean(numeric_only=True).to_dict(),
        "score_oos": score_oos.mean(numeric_only=True).to_dict() if not score_oos.empty else {},
        "score_oos_status": score_oos_status,
        "coverage": coverage.to_dict(orient="records"),
        "archive_audit": archive_audit,
        "pit_policy": PIT_POLICY,
        "pit_verified_rows": pit_verified,
        "pit_total_rows": pit_total,
        "pit_verified_rate": float(pit_verified / pit_total) if pit_total else 0.0,
        "oos_protocol": {
            "development_blocks": int(len(development_oos)),
            "locked_blocks": 2,
            "locked_oos_untouched": True,
            "selection_source": "historical_validation_only",
            "temporal_integrity": oos_temporal,
            "score_temporal_integrity": score_temporal,
        },
        "stability_gate": stability,
        "accuracy_target": {
            "target_accuracy": TARGET_ACCURACY,
            "locked_oos_accuracy": locked_acc,
            "weighted_oos_accuracy": oos_acc,
            "target_met": bool(locked_acc >= TARGET_ACCURACY) if locked_n else False,
            "target_gap": float(locked_acc - TARGET_ACCURACY) if locked_n else float("nan"),
            "oos_sample_size": oos_n,
            "locked_oos_sample_size": locked_n,
        },
        "adoption": adoption,
        "production_model": "calibrated_ensemble" if adoption.get("status") == "ADOPT" else "baseline_logistic",
        "production_model_bundle": model_bundle,
        "audit": audit_report,
        "gates": gates,
        "oos_claimed": oos_claimed,
        "ai_research": weakness_advice(wf.mean(numeric_only=True).to_dict()),
    }
    _write_status(out, report); return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False, default=str))
