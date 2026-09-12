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

    # Source-level diagnostic is cheap and remains full coverage.
    pit_diag = build_pit_diagnostic(history)
    pit_diag.to_csv(out / "pit_diagnostic_7x16.csv", index=False)

    # The full 42k-row PIT replay can require a large number of Wayback
    # snapshots. CI therefore supports a deterministic audit mode. Production
    # research remains full replay when PIT_REPLAY_ROWS_PER_GROUP is unset/0.
    try:
        rows_per_group = max(0, int(os.getenv("PIT_REPLAY_ROWS_PER_GROUP", "0")))
    except ValueError:
        rows_per_group = 0
    replay_input = _pit_sample(history, rows_per_group)
    history_replayed = apply_pit_evidence(replay_input)
    history_replayed["source_available_at_utc"] = pd.to_datetime(history_replayed["source_available_at_utc"], utc=True, errors="coerce")
    history_replayed["retrieved_at_utc"] = pd.to_datetime(history_replayed["retrieved_at_utc"], utc=True, errors="coerce")
    history_replayed.to_csv(out / "normalized_history.csv", index=False)

    feats = build_match_features(history_replayed, history_replayed)
    feats = add_target(feats, history_replayed)
    feats.to_csv(out / "pit_replay_features.csv", index=False)

    verified = feats[feats["pit_verified"] == True].copy()
    if len(verified) < 400:
        report = {
            "status": "BLOCKED",
            "reason": "PIT replay coverage is insufficient; historical result availability cannot yet support the required OOS sample.",
            "replay_mode": "sampled" if rows_per_group else "full",
            "rows_per_competition_season": rows_per_group if rows_per_group else None,
            "acquired_rows": int(len(history)),
            "replayed_rows": int(len(replay_input)),
            "pit_verified_rows": int(len(verified)),
            "pit_verified_rate": float(len(verified) / len(feats)) if len(feats) else 0.0,
            "source_result_verified_rows": int(history_replayed["pit_evidence_status"].eq("VERIFIED").sum()),
            "snapshot_id": snapshot_id(history),
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
        "pit_verified_rows": int(len(verified)),
        "pit_verified_rate": float(len(verified) / len(feats)),
        "ai_research": weakness_advice(summary),
    }
    (out / "run_status.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False, default=str))
