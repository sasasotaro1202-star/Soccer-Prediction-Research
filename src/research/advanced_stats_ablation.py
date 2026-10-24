from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.global_datalake_adapter import load_global_datalake_history
from src.evaluation.walk_forward import run_walk_forward
from src.features.soccer_features import add_target, build_match_features


ADVANCED_FEATURE_TOKENS = (
    "xg_",
    "_xg_",
    "possession_",
    "_possession_",
    "offsides_",
    "_offsides_",
    "pass_accuracy_",
    "_pass_accuracy_",
    "goals_ht_",
    "_goals_ht_",
    "xg_ht_",
    "_xg_ht_",
    "shots_inside_box_",
    "_shots_inside_box_",
    "shots_outside_box_",
    "_shots_outside_box_",
    "blocked_shots_",
    "_blocked_shots_",
    "penalties_",
    "_penalties_",
)


def _model_features(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "match_id", "competition", "season", "season_start", "kickoff_utc",
        "home_team", "away_team", "prediction_cutoff_at_utc",
        "home_goals", "away_goals", "target", "pit_verified",
        "feature_source_max_available_at_utc",
    }
    cols = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    cols = [c for c in cols if c not in excluded and not c.startswith("baseline_")]
    if not cols:
        raise RuntimeError("No numeric model features available")
    return cols


def _is_advanced_feature(name: str) -> bool:
    value = str(name)
    return any(token in value for token in ADVANCED_FEATURE_TOKENS)


def _drop_advanced_features(frame: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in frame.columns if _is_advanced_feature(c)]
    return frame.drop(columns=drop, errors="ignore")


def _summarize(name: str, oos: pd.DataFrame, feature_count: int) -> pd.DataFrame:
    if oos.empty:
        raise RuntimeError(f"{name} OOS output is empty")
    required = {"logloss", "brier", "accuracy", "ece", "oos_start", "oos_end"}
    missing = sorted(required - set(oos.columns))
    if missing:
        raise RuntimeError(f"{name} OOS output missing required columns: {missing}")
    result = oos[["oos_start", "oos_end", "logloss", "brier", "accuracy", "ece"]].copy()
    result.insert(0, "variant", name)
    result["feature_count"] = int(feature_count)
    return result


def run(
    *,
    start_year: int = 2012,
    end_year: int = 2025,
    output_dir: str = "artifacts/advanced_stats_ablation",
    cache_dir: str = "cache/global_datalake_ablation",
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    history, source_status = load_global_datalake_history(
        start_year=start_year,
        end_year=end_year,
        cache_dir=cache_dir,
    )
    if history.empty:
        raise RuntimeError("Global Data Lake returned no mapped PIT history")

    feats = add_target(build_match_features(history, history), history)
    feats = feats[feats["pit_verified"] == True].copy()
    feats = feats.dropna(subset=["target"]).reset_index(drop=True)
    if feats.empty:
        raise RuntimeError("No PIT-verified rows available for ablation")

    full_features = _model_features(feats)
    baseline = _drop_advanced_features(feats)
    baseline_features = _model_features(baseline)

    full_oos, _ = run_walk_forward(feats, full_features)
    baseline_oos, _ = run_walk_forward(baseline, baseline_features)

    full_summary = _summarize("advanced_stats", full_oos, len(full_features))
    baseline_summary = _summarize("baseline_no_advanced_stats", baseline_oos, len(baseline_features))
    comparison = pd.concat([baseline_summary, full_summary], ignore_index=True)
    comparison.to_csv(out / "oos_metrics_by_variant.csv", index=False)

    numeric = comparison[["logloss", "brier", "accuracy", "ece"]].groupby(comparison["variant"]).mean()
    delta = {
        "logloss_delta_advanced_minus_baseline": float(
            numeric.loc["advanced_stats", "logloss"] - numeric.loc["baseline_no_advanced_stats", "logloss"]
        ),
        "brier_delta_advanced_minus_baseline": float(
            numeric.loc["advanced_stats", "brier"] - numeric.loc["baseline_no_advanced_stats", "brier"]
        ),
        "accuracy_delta_advanced_minus_baseline": float(
            numeric.loc["advanced_stats", "accuracy"] - numeric.loc["baseline_no_advanced_stats", "accuracy"]
        ),
        "ece_delta_advanced_minus_baseline": float(
            numeric.loc["advanced_stats", "ece"] - numeric.loc["baseline_no_advanced_stats", "ece"]
        ),
    }
    status = {
        "status": "OK",
        "pit_basis": source_status.get("pit_basis"),
        "source_rows": int(len(history)),
        "pit_verified_rows": int(len(feats)),
        "baseline_feature_count": len(baseline_features),
        "advanced_feature_count": len(full_features),
        "advanced_feature_names": [c for c in full_features if _is_advanced_feature(c)],
        "oos_blocks_baseline": int(len(baseline_oos)),
        "oos_blocks_advanced": int(len(full_oos)),
        "mean_metrics": numeric.to_dict(),
        "delta": delta,
        "production_status": "RESEARCH_ONLY",
    }
    (out / "status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return status


if __name__ == "__main__":
    import os

    result = run(
        start_year=int(os.getenv("START_YEAR", "2012")),
        end_year=int(os.getenv("END_YEAR", "2025")),
    )
    print(json.dumps(result, ensure_ascii=False))
