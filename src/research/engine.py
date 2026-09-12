from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.coverage import build_coverage
from src.data.football_data import load_available_history
from src.data.pit_source_adapter import apply_pit_evidence, build_pit_diagnostic, competition_adapter_matrix
from src.data.source_registry import SOCCER_SOURCES
from src.evaluation.walk_forward import TARGET_ACCURACY, run_walk_forward
from src.features.soccer_features import add_target, build_match_features
from src.research.llm import weakness_advice

EXCLUDED_MODEL_COLUMNS = {
    "match_id", "competition", "season", "season_start", "kickoff_utc", "home_team", "away_team",
    "prediction_cutoff_at_utc", "home_goals", "away_goals", "target", "pit_verified", "feature_source_max_available_at_utc",
}


def snapshot_id(df: pd.DataFrame) -> str:
    excluded = {"retrieved_at_utc", "source_available_at_utc"}
    cols = [c for c in df.columns if c not in excluded]
    stable = df[cols].copy()
    for c in stable.columns:
        if pd.api.types.is_datetime64_any_dtype(stable[c]):
            stable[c] = pd.to_datetime(stable[c], utc=True, errors="coerce").astype("string")
    sort_cols = [c for c in ["competition", "season_start", "kickoff_utc", "home_team", "away_team", "match_id"] if c in stable.columns]
    if sort_cols:
        stable = stable.sort_values(sort_cols, kind="mergesort")
    return hashlib.sha256(stable.to_json(orient="records", date_format="iso").encode()).hexdigest()


def _pit_sample(history: pd.DataFrame, rows_per_group: int) -> pd.DataFrame:
    if rows_per_group <= 0:
        return history.copy()
    h = history.copy()
    sort_cols = [c for c in ["competition", "season_start", "kickoff_utc", "home_team", "away_team"] if c in h.columns]
    if sort_cols:
        h = h.sort_values(sort_cols, kind="mergesort")
    return h.groupby(["competition", "season_start"], sort=False, dropna=False, group_keys=False).head(rows_per_group).copy()


def _write_source_registry(out: Path) -> None:
    pd.DataFrame([
        {"name": s.name, "kind": s.kind, "role": s.role, "fields": ",".join(s.fields),
         "historical": s.historical, "pit_capable": s.pit_capable, "live_capable": s.live_capable,
         "auth_required": s.auth_required, "primary_for": ",".join(s.primary_for), "notes": s.notes}
        for s in SOCCER_SOURCES
    ]).to_csv(out / "source_registry.csv", index=False)


def _model_features(feats: pd.DataFrame) -> list[str]:
    numeric = feats.select_dtypes(include=["number", "bool"]).columns.tolist()
    cols = [c for c in numeric if c not in EXCLUDED_MODEL_COLUMNS]
    if not cols:
        raise ValueError("No numeric model features available")
    return cols


def run(out_dir: str = "artifacts") -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    _write_source_registry(out)
    competition_adapter_matrix().to_csv(out / "pit_competition_adapter_matrix.csv", index=False)
    history, acquisition = load_available_history()
    acquisition.to_csv(out / "acquisition_coverage.csv", index=False)
    coverage = build_coverage(history)
    coverage.to_csv(out / "coverage_matrix.csv", index=False)
    if history.empty:
        report = {"status": "BLOCKED", "reason": "No historical data acquired"}
        (out / "run_status.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    try:
        rows_per_group = max(1, int(os.getenv("PIT_REPLAY_ROWS_PER_GROUP", "1")))
    except ValueError:
        rows_per_group = 1
    replay_input = _pit_sample(history, rows_per_group)
    archive_audit = {"status": "SKIPPED"}
    try:
        pit_diag = build_pit_diagnostic(replay_input)
        pit_diag.to_csv(out / "pit_diagnostic_sample.csv", index=False)
        history_replayed = apply_pit_evidence(replay_input)
        if os.getenv("PIT_ENABLE_SECONDARY_ARCHIVE", "1") == "1":
            try:
                from src.data.pit_archive_fallback import apply_arquivo_fallback
                history_replayed = apply_arquivo_fallback(history_replayed)
            except Exception as exc:
                history_replayed["secondary_archive_error"] = f"{type(exc).__name__}: {exc}"
        archive_audit = {
            "status": "COMPLETED", "rows": int(len(history_replayed)),
            "pit_evidence_status_counts": history_replayed.get("pit_evidence_status", pd.Series(dtype=str)).astype(str).value_counts().to_dict(),
        }
        history_replayed.to_csv(out / "normalized_history_archive_audit.csv", index=False)
    except Exception as exc:
        archive_audit = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}

    feats = build_match_features(history, history)
    feats = add_target(feats, history)
    feats.to_csv(out / "pit_replay_features.csv", index=False)
    pit_verified = int(feats["pit_verified"].sum()) if "pit_verified" in feats.columns else 0
    pit_total = int(len(feats))
    if pit_verified == 0:
        report = {
            "status": "BLOCKED", "reason": "No match rows have sufficient historical result state under the deterministic PIT policy.",
            "acquired_rows": int(len(history)), "snapshot_id": snapshot_id(history), "pit_policy": "result_plus_24h",
            "pit_verified_rows": 0, "pit_verified_rate": 0.0, "archive_audit": archive_audit,
        }
        report["ai_research"] = weakness_advice(report)
        (out / "run_status.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return report

    feature_cols = _model_features(feats)
    wf, selections = run_walk_forward(feats, feature_cols)
    wf.to_csv(out / "oos_metrics.csv", index=False)
    selections.to_csv(out / "model_selection.csv", index=False)
    summary = wf.mean(numeric_only=True).to_dict()
    oos_n = int(wf["n"].sum()) if "n" in wf.columns else 0
    oos_acc = float(np.average(wf["accuracy"], weights=wf["n"])) if oos_n else float("nan")
    report = {
        "status": "OK", "snapshot_id": snapshot_id(history), "oos": summary,
        "coverage": coverage.to_dict(orient="records"), "archive_audit": archive_audit,
        "pit_policy": "result_plus_24h", "pit_verified_rows": pit_verified, "pit_total_rows": pit_total,
        "pit_verified_rate": float(pit_verified / pit_total) if pit_total else 0.0,
        "accuracy_target": {"target_accuracy": TARGET_ACCURACY, "locked_oos_accuracy": oos_acc,
                            "target_met": bool(oos_acc >= TARGET_ACCURACY) if oos_n else False,
                            "target_gap": float(oos_acc - TARGET_ACCURACY) if oos_n else float("nan"),
                            "oos_sample_size": oos_n},
        "ai_research": weakness_advice(summary),
    }
    (out / "run_status.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False, default=str))
