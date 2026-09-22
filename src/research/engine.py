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


def _primary_score_metrics_finite(score_oos: pd.DataFrame, *, min_blocks: int = 3) -> bool:
    if score_oos.empty or len(score_oos) < int(min_blocks):
        return False
    missing = [column for column in PRIMARY_SCORE_METRICS if column not in score_oos.columns]
    if missing:
        return False
    values = score_oos[list(PRIMARY_SCORE_METRICS)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return False
    if "n" not in score_oos.columns:
        return False
    n = pd.to_numeric(score_oos["n"], errors="coerce").to_numpy(dtype=float)
    return bool(np.isfinite(n).all() and (n > 0).all())


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
    oos_ok = bool(
        len(wf) >= 3
        and len(development_oos) >= 1
        and len(locked_oos) == 2
        and score_oos_gate.get("status") == "PASS"
        and int(score_oos_gate.get("blocks", 0)) >= 3
        and score_oos_gate.get("finite_metrics") is True
        and score_locked_gate.get("status") == "PASS"
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
    feats = add_target(build_match_features(history, history), history); feats.to_csv(out / "pit_replay_features.csv", index=False)
    pit_verified = int(feats["pit_verified"].sum()) if "pit_verified" in feats.columns else 0; pit_total = int(len(feats))
    if pit_verified == 0:
        report = {"status": "BLOCKED", "reason": "No match rows have sufficient historical result state under deterministic PIT.", "acquired_rows": int(len(history)), "snapshot_id": snapshot_id(history), "pit_policy": PIT_POLICY, "pit_verified_rows": 0, "pit_verified_rate": 0.0, "archive_audit": archive_audit, "audit": audit_report, "oos_claimed": False}; report["ai_research"] = weakness_advice(report); _write_status(out, report); return report
    try:
        score_oos = run_score_walk_forward(
            feats,
            min_train=max(500, int(os.getenv("SOCCER_SCORE_MIN_TRAIN", "1000"))),
            oos_block=max(500, int(os.getenv("SOCCER_SCORE_OOS_BLOCK", "2000"))),
        )
        score_oos.to_csv(out / "score_oos_metrics.csv", index=False)
        primary_finite = _primary_score_metrics_finite(score_oos)
        score_oos_status = {
            "status": "PASS" if primary_finite else "ERROR",
            "blocks": int(len(score_oos)),
            "rows": int(score_oos["n"].sum()) if "n" in score_oos.columns else 0,
            "finite_metrics": primary_finite,
            "primary_metrics_finite": primary_finite,
            "recency_status": sorted(set(score_oos["recency_status"].astype(str))) if "recency_status" in score_oos.columns else [],
            "dc_status": sorted(set(score_oos["dc_status"].astype(str))) if "dc_status" in score_oos.columns else [],
        }
    except Exception as exc:
        score_oos = pd.DataFrame()
        score_oos_status = {
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "blocks": 0,
            "rows": 0,
            "finite_metrics": False,
        }
    (out / "score_oos_gate.json").write_text(
        json.dumps(score_oos_status, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    score_protocol_ready = len(score_oos) >= 3 and primary_finite
    score_development_oos = score_oos.iloc[:-2].copy() if score_protocol_ready else pd.DataFrame()
    score_selection = select_score_model(score_development_oos)
    score_locked_oos = score_oos.tail(2).copy() if score_protocol_ready else pd.DataFrame()
    score_selection["protocol"] = {
        "minimum_total_blocks": 3,
        "development_blocks": int(len(score_development_oos)),
        "locked_blocks": int(len(score_locked_oos)),
        "locked_oos_untouched_for_selection": True,
    }
    score_locked_gate = verify_selected_score_model(score_selection, score_locked_oos)
    (out / "score_model_selection.json").write_text(
        json.dumps(score_selection, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (out / "score_locked_gate.json").write_text(
        json.dumps(score_locked_gate, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    wf, selections = run_walk_forward(feats, _model_features(feats)); wf.to_csv(out / "oos_metrics.csv", index=False); selections.to_csv(out / "model_selection.csv", index=False)
    if len(wf) < 3:
        report = {"status": "BLOCKED", "reason": "At least three chronological OOS blocks are required: development plus two locked holdout blocks.", "snapshot_id": snapshot_id(history), "pit_policy": PIT_POLICY, "pit_verified_rows": pit_verified, "pit_total_rows": pit_total, "archive_audit": archive_audit, "audit": audit_report, "oos_claimed": False}; _write_status(out, report); return report
    development_oos = wf.iloc[:-2].copy(); locked = wf.tail(2).copy(); development_oos.to_csv(out / "development_oos_metrics.csv", index=False); locked.to_csv(out / "locked_oos_metrics.csv", index=False)
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
    adoption = adoption_decision(baseline, candidate, development_oos=development_oos, min_accuracy=TARGET_ACCURACY)
    adoption["external_stability_gate"] = stability
    (out / "adoption_decision.json").write_text(json.dumps(adoption, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    model_bundle = None
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
                },
                training_end=model_bundle.get("fit_end"),
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
