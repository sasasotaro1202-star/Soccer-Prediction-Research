from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data.coverage import build_coverage
from src.data.football_data import load_available_history
from src.data.pit_source_adapter import apply_pit_evidence, competition_adapter_matrix
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

    # Establish when each historical result first became observable in the
    # source. This timestamp is about the past result itself, not the target
    # match being predicted.
    history = apply_pit_evidence(history)
    history["source_available_at_utc"] = pd.to_datetime(history["source_available_at_utc"], utc=True, errors="coerce")
    history["retrieved_at_utc"] = pd.to_datetime(history["retrieved_at_utc"], utc=True, errors="coerce")
    history.to_csv(out / "normalized_history.csv", index=False)

    # Chronological replay: build every prediction row from past matches only;
    # the feature builder rejects any rolling window containing a result whose
    # source was not available by that prediction cutoff.
    feats = build_match_features(history, history)
    feats = add_target(feats, history)
    feats.to_csv(out / "pit_replay_features.csv", index=False)

    verified = feats[feats["pit_verified"] == True].copy()
    if len(verified) < 400:
        report = {
            "status": "BLOCKED",
            "reason": "PIT replay coverage is insufficient; historical result availability cannot yet support the required OOS sample.",
            "acquired_rows": int(len(history)),
            "pit_verified_rows": int(len(verified)),
            "pit_verified_rate": float(len(verified) / len(feats)) if len(feats) else 0.0,
            "source_result_verified_rows": int(history["pit_evidence_status"].eq("VERIFIED").sum()),
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
