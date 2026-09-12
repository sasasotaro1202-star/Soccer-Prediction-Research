from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from src.data.coverage import build_coverage
from src.data.football_data import load_available_history
from src.data.pit_source_adapter import apply_pit_evidence, build_pit_diagnostic, competition_adapter_matrix
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
    payload = df.to_json(orient="records", date_format="iso")
    return hashlib.sha256(payload.encode()).hexdigest()


def _pit_sample(history: pd.DataFrame, rows_per_group: int) -> pd.DataFrame:
    """Deterministic PIT sample: earliest chronological rows per competition/season."""
    if rows_per_group <= 0:
        return history.copy()
    h = history.copy()
    sort_cols = [c for c in ["competition", "season_start", "kickoff_utc", "home_team", "away_team"] if c in h.columns]
    if sort_cols:
        h = h.sort_values(sort_cols, kind="mergesort")
    return h.groupby(["competition", "season_start"], sort=False, dropna=False, group_keys=False).head(rows_per_group).copy()


def _pit_gate(history_replayed: pd.DataFrame, replay_input: pd.DataFrame) -> tuple[bool, dict]:
    """Hard PIT gate: every replayed row must have auditable evidence.

    A partial PIT sample is never sufficient for model research. In particular,
    this prevents the research engine from silently training on a mixture of
    verified and unverifiable historical outcomes.
    """
    total = int(len(replay_input))
    verified = int(history_replayed["pit_evidence_status"].eq("VERIFIED").sum()) if "pit_evidence_status" in history_replayed else 0
    unverifiable = total - verified
    reasons = {}
    if "pit_evidence_reason" in history_replayed:
        reasons = {str(k): int(v) for k, v in history_replayed["pit_evidence_reason"].fillna("").value_counts().items()}
    return total > 0 and verified == total, {
        "replayed_rows": total,
        "pit_verified_rows": verified,
        "pit_unverifiable_rows": unverifiable,
        "pit_verified_rate": float(verified / total) if total else 0.0,
        "pit_reason_counts": reasons,
    }


def run(out_dir: str = "artifacts") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    competition_adapter_matrix().to_csv(out / "pit_competition_adapter_matrix.csv", index=False)

    history, acquisition = load_available_history()
    acquisition.to_csv(out / "acquisition_coverage.csv", index=False)
    coverage = build_coverage(history)
    coverage.to_csv(out / "coverage_matrix.csv", index=False)
    if history.empty:
        report = {"status": "BLOCKED", "reason": "No historical data acquired"}
        (out / "run_status.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    pit_diag = build_pit_diagnostic(history)
    pit_diag.to_csv(out / "pit_diagnostic_7x16.csv", index=False)

    try:
        rows_per_group = max(0, int(os.getenv("PIT_REPLAY_ROWS_PER_GROUP", "0")))
    except ValueError:
        rows_per_group = 0
    replay_input = _pit_sample(history, rows_per_group)
    history_replayed = apply_pit_evidence(replay_input)

    # Secondary PIT evidence is intentionally isolated from the engine's core
    # adapter. It may be enabled explicitly for audit/research, but every row
    # still needs an exact archived-result match and timestamp.
    if os.getenv("PIT_ENABLE_SECONDARY_ARCHIVE", "1") == "1":
        try:
            from src.data.pit_archive_fallback import apply_arquivo_fallback
            history_replayed = apply_arquivo_fallback(history_replayed)
        except Exception as exc:
            # Fail closed: a broken secondary provider cannot create evidence.
            history_replayed["secondary_archive_error"] = f"{type(exc).__name__}: {exc}"

    history_replayed["source_available_at_utc"] = pd.to_datetime(history_replayed["source_available_at_utc"], utc=True, errors="coerce")
    history_replayed["retrieved_at_utc"] = pd.to_datetime(history_replayed["retrieved_at_utc"], utc=True, errors="coerce")
    history_replayed.to_csv(out / "normalized_history.csv", index=False)

    pit_ok, pit_summary = _pit_gate(history_replayed, replay_input)
    if not pit_ok:
        report = {
            "status": "BLOCKED",
            "reason": "PIT hard gate failed; research/OOS is forbidden until every replayed row has exact archive evidence.",
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
        "status": "OK",
        "snapshot_id": snapshot_id(history),
        "oos": summary,
        "coverage": coverage.to_dict(orient="records"),
        **pit_summary,
        "ai_research": weakness_advice(summary),
    }
    (out / "run_status.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False, default=str))
