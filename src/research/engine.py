from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from src.data.coverage import build_coverage
from src.data.football_data import load_available_history
from src.data.pit_policy import result_feature_available_at
from src.data.pit_source_adapter import apply_pit_evidence, build_pit_diagnostic, competition_adapter_matrix
from src.data.source_registry import SOCCER_SOURCES
from src.evaluation.walk_forward import run_walk_forward
from src.features.soccer_features import add_target, build_match_features
from src.research.llm import weakness_advice

FEATURES = [
    "home_gf_3", "home_ga_3", "home_points_3", "home_gd_3", "away_gf_3", "away_ga_3", "away_points_3", "away_gd_3",
    "home_gf_5", "home_ga_5", "home_points_5", "home_gd_5", "away_gf_5", "away_ga_5", "away_points_5", "away_gd_5",
    "home_gf_10", "home_ga_10", "home_points_10", "home_gd_10", "away_gf_10", "away_ga_10", "away_points_10", "away_gd_10",
    "home_gd_5_minus_away_gd_5", "home_points_5_minus_away_points_5", "home_advantage",
]


def snapshot_id(df: pd.DataFrame) -> str:
    """Hash only data-bearing fields so operational retrieval time cannot change the snapshot ID."""
    excluded = {"retrieved_at_utc", "source_available_at_utc"}
    cols = [c for c in df.columns if c not in excluded]
    stable = df[cols].copy()
    for c in stable.columns:
        if pd.api.types.is_datetime64_any_dtype(stable[c]):
            stable[c] = pd.to_datetime(stable[c], utc=True, errors="coerce").astype("string")
    stable = stable.sort_values([c for c in ["competition", "season", "kickoff_utc", "home_team", "away_team", "match_id"] if c in stable.columns], kind="mergesort")
    return hashlib.sha256(stable.to_json(orient="records", date_format="iso").encode()).hexdigest()


def _pit_sample(history: pd.DataFrame, rows_per_group: int) -> pd.DataFrame:
    """Deterministic PIT sample: earliest chronological rows per competition/season."""
    if rows_per_group <= 0:
        return history.copy()
    h = history.copy()
    sort_cols = [c for c in ["competition", "season_start", "kickoff_utc", "home_team", "away_team"] if c in h.columns]
    if sort_cols:
        h = h.sort_values(sort_cols, kind="mergesort")
    return h.groupby(["competition", "season_start"], sort=False, dropna=False, group_keys=False).head(rows_per_group).copy()


def _policy_pit_gate(history_replayed: pd.DataFrame, replay_input: pd.DataFrame) -> tuple[bool, dict]:
    """Fast deterministic gate for result-derived features.

    Archive evidence is retained as an audit channel, but an archive transport failure
    is not allowed to masquerade as a future-data leak. A row is usable only when every
    historical result used by the feature policy is at least 24h old at the cutoff.
    """
    total = int(len(replay_input))
    if total == 0:
        return False, {"replayed_rows": 0, "pit_verified_rows": 0, "pit_unverifiable_rows": 0, "pit_verified_rate": 0.0}
    kickoff = pd.to_datetime(replay_input["kickoff_utc"], utc=True, errors="coerce")
    cutoff = kickoff - pd.Timedelta(minutes=60)
    available = result_feature_available_at(kickoff)
    verified = available.notna() & cutoff.notna() & available.le(cutoff)
    reasons = {}
    if "pit_evidence_status" in history_replayed.columns:
        archive_status = history_replayed["pit_evidence_status"].astype(str)
        reasons = {str(k): int(v) for k, v in archive_status.value_counts().items()}
    n = int(verified.sum())
    return n == total, {
        "replayed_rows": total,
        "pit_verified_rows": n,
        "pit_unverifiable_rows": total - n,
        "pit_verified_rate": float(n / total),
        "pit_policy": "result_plus_24h",
        "archive_audit_status_counts": reasons,
    }


def _write_source_registry(out: Path) -> None:
    pd.DataFrame([
        {
            "name": s.name, "kind": s.kind, "role": s.role, "fields": ",".join(s.fields),
            "historical": s.historical, "pit_capable": s.pit_capable, "live_capable": s.live_capable,
            "auth_required": s.auth_required, "primary_for": ",".join(s.primary_for), "notes": s.notes,
        }
        for s in SOCCER_SOURCES
    ]).to_csv(out / "source_registry.csv", index=False)


def run(out_dir: str = "artifacts") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
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
        rows_per_group = max(0, int(os.getenv("PIT_REPLAY_ROWS_PER_GROUP", "0")))
    except ValueError:
        rows_per_group = 0
    replay_input = _pit_sample(history, rows_per_group)

    # Expensive archive diagnostics are performed only on the deterministic PIT sample,
    # never on all 42k+ rows. This preserves auditability while removing the main runtime sink.
    pit_diag = build_pit_diagnostic(replay_input)
    pit_diag.to_csv(out / "pit_diagnostic_sample.csv", index=False)
    history_replayed = apply_pit_evidence(replay_input)

    if os.getenv("PIT_ENABLE_SECONDARY_ARCHIVE", "1") == "1":
        try:
            from src.data.pit_archive_fallback import apply_arquivo_fallback
            history_replayed = apply_arquivo_fallback(history_replayed)
        except Exception as exc:
            history_replayed["secondary_archive_error"] = f"{type(exc).__name__}: {exc}"

    for col in ("source_available_at_utc", "retrieved_at_utc"):
        if col in history_replayed.columns:
            history_replayed[col] = pd.to_datetime(history_replayed[col], utc=True, errors="coerce")
    history_replayed.to_csv(out / "normalized_history.csv", index=False)

    pit_ok, pit_summary = _policy_pit_gate(history_replayed, replay_input)
    if not pit_ok:
        report = {
            "status": "BLOCKED",
            "reason": "PIT policy gate failed; required result-availability lag is not satisfied.",
            "replay_mode": "sampled" if rows_per_group else "full",
            "rows_per_competition_season": rows_per_group if rows_per_group else None,
            "acquired_rows": int(len(history)),
            "snapshot_id": snapshot_id(history),
            **pit_summary,
        }
        report["ai_research"] = weakness_advice(report)
        (out / "run_status.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return report

    feats = build_match_features(history_replayed, history_replayed)
    feats = add_target(feats, history_replayed)
    feats.to_csv(out / "pit_replay_features.csv", index=False)

    if len(feats) < 400:
        report = {
            "status": "BLOCKED",
            "reason": "PIT evidence passed, but the verified sample is too small for the required OOS research sample.",
            "replay_mode": "sampled" if rows_per_group else "full",
            "acquired_rows": int(len(history)),
            "snapshot_id": snapshot_id(history),
            **pit_summary,
        }
        report["ai_research"] = weakness_advice(report)
        (out / "run_status.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return report

    wf, selections = run_walk_forward(feats, FEATURES)
    wf.to_csv(out / "oos_metrics.csv", index=False)
    selections.to_csv(out / "model_selection.csv", index=False)
    summary = wf.mean(numeric_only=True).to_dict()
    report = {
        "status": "OK", "snapshot_id": snapshot_id(history), "oos": summary,
        "coverage": coverage.to_dict(orient="records"), **pit_summary,
        "ai_research": weakness_advice(summary),
    }
    (out / "run_status.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False, default=str))
