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
from src.evaluation.walk_forward import TARGET_ACCURACY, run_walk_forward
from src.features.soccer_features import add_target, build_match_features
from src.research.adoption import adoption_decision
from src.research.llm import weakness_advice

EXCLUDED_MODEL_COLUMNS = {"match_id", "competition", "season", "season_start", "kickoff_utc", "home_team", "away_team", "prediction_cutoff_at_utc", "home_goals", "away_goals", "target", "pit_verified", "feature_source_max_available_at_utc"}
PIT_POLICY = "explicit_source_publication_time_only; unknown_publication_time_excluded"


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


def run(out_dir: str = "artifacts") -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); _write_source_registry(out)
    try:
        audit_report = run_audit(str(out))
    except Exception as exc:
        audit_report = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}", "audit_complete": False}
    (out / "audit_gate.json").write_text(json.dumps(audit_report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    competition_adapter_matrix().to_csv(out / "pit_competition_adapter_matrix.csv", index=False)
    history, acquisition = load_available_history(); acquisition.to_csv(out / "acquisition_coverage.csv", index=False)
    coverage = build_coverage(history); coverage.to_csv(out / "coverage_matrix.csv", index=False)
    if history.empty:
        report = {"status": "BLOCKED", "reason": "No historical data acquired", "oos_claimed": False, "audit": audit_report}; _write_status(out, report); return report
    archive_audit = _archive_audit_sample(history, out)
    feats = add_target(build_match_features(history, history), history); feats.to_csv(out / "pit_replay_features.csv", index=False)
    pit_verified = int(feats["pit_verified"].sum()) if "pit_verified" in feats.columns else 0; pit_total = int(len(feats))
    if pit_verified == 0:
        report = {"status": "BLOCKED", "reason": "No match rows have sufficient historical result state under deterministic PIT.", "acquired_rows": int(len(history)), "snapshot_id": snapshot_id(history), "pit_policy": PIT_POLICY, "pit_verified_rows": 0, "pit_verified_rate": 0.0, "archive_audit": archive_audit, "audit": audit_report, "oos_claimed": False}; report["ai_research"] = weakness_advice(report); _write_status(out, report); return report
    wf, selections = run_walk_forward(feats, _model_features(feats)); wf.to_csv(out / "oos_metrics.csv", index=False); selections.to_csv(out / "model_selection.csv", index=False)
    if len(wf) < 3:
        report = {"status": "BLOCKED", "reason": "At least three chronological OOS blocks are required: development plus two locked holdout blocks.", "snapshot_id": snapshot_id(history), "pit_policy": PIT_POLICY, "pit_verified_rows": pit_verified, "pit_total_rows": pit_total, "archive_audit": archive_audit, "audit": audit_report, "oos_claimed": False}; _write_status(out, report); return report
    development_oos = wf.iloc[:-2].copy(); locked = wf.tail(2).copy(); development_oos.to_csv(out / "development_oos_metrics.csv", index=False); locked.to_csv(out / "locked_oos_metrics.csv", index=False)
    candidate_lock = {"status": "LOCKED", "selection_source": "historical_validation_only", "selection_artifact": "model_selection.csv", "development_oos_blocks": int(len(development_oos)), "locked_oos_blocks": int(len(locked)), "locked_oos_untouched": True, "target_accuracy": TARGET_ACCURACY, "model_family": "validation-selected calibrated ensemble", "feature_policy": "PIT-safe numeric features only", "pit_policy": PIT_POLICY}
    (out / "candidate_lock.json").write_text(json.dumps(candidate_lock, indent=2, ensure_ascii=False), encoding="utf-8")
    baseline_cols = ["oos_start", "oos_end", "baseline_logistic_logloss", "baseline_logistic_accuracy", "baseline_logistic_brier", "baseline_logistic_rps", "baseline_logistic_ece", "n"]
    candidate_cols = ["oos_start", "oos_end", "logloss", "accuracy", "brier", "rps", "ece", "n"]
    baseline = locked[baseline_cols].rename(columns={"baseline_logistic_logloss": "logloss", "baseline_logistic_accuracy": "accuracy", "baseline_logistic_brier": "brier", "baseline_logistic_rps": "rps", "baseline_logistic_ece": "ece"})
    candidate = locked[candidate_cols].copy()
    adoption = adoption_decision(baseline, candidate, development_oos=development_oos, min_accuracy=TARGET_ACCURACY)
    (out / "adoption_decision.json").write_text(json.dumps(adoption, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    oos_n = int(wf["n"].sum()) if "n" in wf.columns else 0; oos_acc = float(np.average(wf["accuracy"], weights=wf["n"])) if oos_n else float("nan")
    locked_n = int(locked["n"].sum()); locked_acc = float(np.average(locked["accuracy"], weights=locked["n"])) if locked_n else float("nan")
    report = {"status": "OK", "snapshot_id": snapshot_id(history), "oos": wf.mean(numeric_only=True).to_dict(), "coverage": coverage.to_dict(orient="records"), "archive_audit": archive_audit, "pit_policy": PIT_POLICY, "pit_verified_rows": pit_verified, "pit_total_rows": pit_total, "pit_verified_rate": float(pit_verified / pit_total) if pit_total else 0.0, "oos_protocol": {"development_blocks": int(len(development_oos)), "locked_blocks": 2, "locked_oos_untouched": True, "selection_source": "historical_validation_only"}, "accuracy_target": {"target_accuracy": TARGET_ACCURACY, "locked_oos_accuracy": locked_acc, "weighted_oos_accuracy": oos_acc, "target_met": bool(locked_acc >= TARGET_ACCURACY) if locked_n else False, "target_gap": float(locked_acc - TARGET_ACCURACY) if locked_n else float("nan"), "oos_sample_size": oos_n, "locked_oos_sample_size": locked_n}, "adoption": adoption, "production_model": "calibrated_ensemble" if adoption.get("status") == "ADOPT" else "baseline_logistic", "audit": audit_report, "ai_research": weakness_advice(wf.mean(numeric_only=True).to_dict())}
    _write_status(out, report); return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False, default=str))
