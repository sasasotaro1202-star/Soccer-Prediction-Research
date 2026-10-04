from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.walk_forward import run_walk_forward
from src.research.feature_set_variants import (
    select_feature_set,
    variant_catalog,
)


BASE_MODELS = (
    "elo_logistic",
    "dynamic_elo_logistic",
    "logistic",
    "logistic_select",
    "hist_gb",
)

MODEL_ECOLOGY = (
    "elo_logistic",
    "dynamic_elo_logistic",
    "hierarchical_elo_logistic",
    "recency_logistic",
    "recency_half_life_300",
    "recency_half_life_1200",
    "quantile_logistic",
    "quantile_32",
    "quantile_128",
    "logistic",
    "logistic_c0_15",
    "logistic_c2",
    "logistic_c4",
    "logistic_select",
    "logistic_l2_strong",
    "extra_trees",
    "extra_trees_600",
    "random_forest",
    "random_forest_600",
    "hist_gb",
    "hist_gb_robust",
    "hist_gb_deep",
    "hist_gb_high_reg",
)

CALIBRATION_MODES = ("none", "global", "context", "full")
ROUTING_MODES = ("global", "context", "dynamic")
TRAINING_WINDOWS = (
    ("expanding", None),
    ("rolling_8000", 8000),
    ("rolling_12000", 12000),
)

LOCKED_BLOCKS = 2
CONFIRM_BLOCKS = 2


def _validate_input(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "match_id",
        "kickoff_utc",
        "target",
        "pit_verified",
        "prediction_cutoff_at_utc",
        "feature_source_max_available_at_utc",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Ultimate matrix input missing required columns: {missing}")

    frame["kickoff_utc"] = pd.to_datetime(frame["kickoff_utc"], utc=True, errors="coerce")
    frame["prediction_cutoff_at_utc"] = pd.to_datetime(
        frame["prediction_cutoff_at_utc"], utc=True, errors="coerce"
    )
    frame["feature_source_max_available_at_utc"] = pd.to_datetime(
        frame["feature_source_max_available_at_utc"], utc=True, errors="coerce"
    )
    if frame["kickoff_utc"].isna().any() or frame["prediction_cutoff_at_utc"].isna().any():
        raise RuntimeError("Ultimate matrix input contains invalid temporal fields")

    pit = frame["pit_verified"].astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0", "yes", "no"}
    if not pit.isin(allowed).all():
        raise RuntimeError("Ultimate matrix input contains ambiguous pit_verified values")
    frame["pit_verified"] = pit.isin({"true", "1", "yes"})

    valid = frame["pit_verified"]
    cutoff_ok = frame["feature_source_max_available_at_utc"] <= frame["prediction_cutoff_at_utc"]
    cutoff_ok &= frame["prediction_cutoff_at_utc"] <= frame["kickoff_utc"]
    if not bool((valid & cutoff_ok).all()):
        bad = int((valid & cutoff_ok).eq(False).sum())
        raise RuntimeError(f"Ultimate matrix PIT validation failed for {bad} rows")

    frame = frame.loc[valid].sort_values(
        ["kickoff_utc", "match_id"], kind="mergesort"
    ).reset_index(drop=True)
    return frame


def _weighted(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return float("nan")
    v = pd.to_numeric(frame[column], errors="coerce")
    w = pd.to_numeric(frame["n"], errors="coerce")
    mask = v.notna() & w.notna() & (w > 0)
    if not bool(mask.any()):
        return float("nan")
    return float(np.average(v.loc[mask], weights=w.loc[mask]))


def summarize_oos(wf: pd.DataFrame) -> dict:
    if wf.empty:
        raise RuntimeError("Cannot summarize empty WFO result")
    blocks = len(wf)
    locked_n = min(LOCKED_BLOCKS, blocks)
    dev = wf.iloc[: max(0, blocks - locked_n)]
    locked = wf.tail(locked_n)

    confirm_n = min(CONFIRM_BLOCKS, len(dev))
    screen = dev.iloc[:-confirm_n] if confirm_n else dev
    confirm = dev.tail(confirm_n) if confirm_n else dev.iloc[0:0]

    return {
        "blocks": int(blocks),
        "development_blocks": int(len(dev)),
        "screen_blocks": int(len(screen)),
        "confirm_blocks": int(len(confirm)),
        "locked_blocks": int(len(locked)),
        "development_n": int(pd.to_numeric(dev["n"]).sum()) if not dev.empty else 0,
        "screen_n": int(pd.to_numeric(screen["n"]).sum()) if not screen.empty else 0,
        "confirm_n": int(pd.to_numeric(confirm["n"]).sum()) if not confirm.empty else 0,
        "locked_n": int(pd.to_numeric(locked["n"]).sum()) if not locked.empty else 0,
        "development_logloss": _weighted(dev, "logloss"),
        "screen_logloss": _weighted(screen, "logloss"),
        "confirm_logloss": _weighted(confirm, "logloss"),
        "locked_logloss": _weighted(locked, "logloss"),
        "development_brier": _weighted(dev, "brier"),
        "screen_brier": _weighted(screen, "brier"),
        "confirm_brier": _weighted(confirm, "brier"),
        "locked_brier": _weighted(locked, "brier"),
        "development_accuracy": _weighted(dev, "accuracy"),
        "screen_accuracy": _weighted(screen, "accuracy"),
        "confirm_accuracy": _weighted(confirm, "accuracy"),
        "locked_accuracy": _weighted(locked, "accuracy"),
        "development_ece": _weighted(dev, "ece"),
        "screen_ece": _weighted(screen, "ece"),
        "confirm_ece": _weighted(confirm, "ece"),
        "locked_ece": _weighted(locked, "ece"),
        "oos_window_signature": str(wf["oos_window_signature"].iloc[0]),
    }


def _rank(rows: pd.DataFrame, score_column: str, group_cols: list[str]) -> pd.DataFrame:
    if rows.empty:
        raise RuntimeError("No research rows to rank")
    parts = []
    for _, group in rows.groupby(group_cols, dropna=False, sort=False):
        ordered = group.sort_values(
            [score_column, "confirm_brier", "confirm_ece"],
            kind="mergesort",
        )
        parts.append(ordered.iloc[0])
    return pd.DataFrame(parts).reset_index(drop=True)


def run_ultimate_matrix(
    input_path: str,
    output_dir: str,
    *,
    min_train: int = 2000,
    oos_block: int = 4000,
    random_state: int = 42,
    top_features: int = 6,
    top_combos: int = 3,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = _validate_input(input_path)

    features = [row["variant"] for row in variant_catalog()]
    catalog = {
        "schema_version": 1,
        "phases": {
            "feature_screen": {
                "variants": features,
                "models": list(BASE_MODELS),
                "training": "expanding",
                "calibration": "full",
                "routing": "dynamic",
                "purpose": "feature_family_window_representation_screen",
            },
            "model_screen": {
                "feature_candidates": int(top_features),
                "models": list(MODEL_ECOLOGY),
                "prediction_mode": "fixed:model",
                "purpose": "model_architecture_and_hyperparameter_screen",
            },
            "configuration_screen": {
                "combo_candidates": int(top_combos),
                "training_windows": [name for name, _ in TRAINING_WINDOWS],
                "calibration": list(CALIBRATION_MODES),
                "routing": list(ROUTING_MODES),
                "purpose": "training_window_calibration_routing_screen",
            },
            "locked_blocks": LOCKED_BLOCKS,
            "selection_rule": "screen_then_confirm_then_locked",
            "pit_required": True,
            "production_changed": False,
        },
    }
    (out / "ultimate_matrix_catalog.json").write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    feature_rows = []
    feature_wfs: dict[str, pd.DataFrame] = {}
    for variant in features:
        cols, meta = select_feature_set(frame, variant)
        wf, _ = run_walk_forward(
            frame,
            cols,
            min_train=min_train,
            oos_block=oos_block,
            random_state=random_state,
            candidate_names=list(BASE_MODELS),
            calibration_mode="full",
            routing_mode="dynamic",
            prediction_mode="ensemble",
        )
        summary = summarize_oos(wf)
        feature_rows.append({**meta, **summary})
        feature_wfs[variant] = wf

    feature_table = pd.DataFrame(feature_rows)
    feature_table.to_csv(out / "ultimate_feature_screen.csv", index=False)
    window_signatures = feature_table["oos_window_signature"].dropna().unique().tolist()
    if len(window_signatures) != 1:
        raise RuntimeError("Feature screen variants do not share one OOS window signature")

    selected_features = (
        feature_table.sort_values(
            ["screen_logloss", "screen_brier", "screen_ece", "variant"],
            kind="mergesort",
        )
        .head(int(top_features))["variant"]
        .tolist()
    )
    (out / "ultimate_selected_features.json").write_text(
        json.dumps(selected_features, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    model_rows = []
    for variant in selected_features:
        cols, meta = select_feature_set(frame, variant)
        for model_name in MODEL_ECOLOGY:
            anchor = "logistic" if model_name != "logistic" else "logistic_c0_15"
            wf, _ = run_walk_forward(
                frame,
                cols,
                min_train=min_train,
                oos_block=oos_block,
                random_state=random_state,
                candidate_names=[model_name, anchor],
                calibration_mode="full",
                routing_mode="dynamic",
                prediction_mode=f"fixed:{model_name}",
            )
            summary = summarize_oos(wf)
            model_rows.append({
                **meta,
                "model_name": model_name,
                "anchor_model": anchor,
                **summary,
            })

    model_table = pd.DataFrame(model_rows)
    model_table.to_csv(out / "ultimate_model_screen.csv", index=False)
    model_best = model_table.sort_values(
        ["confirm_logloss", "confirm_brier", "confirm_ece", "variant", "model_name"],
        kind="mergesort",
    )
    selected_combos = model_best.head(int(top_combos))[
        ["variant", "feature_set_id", "model_name", "oos_window_signature"]
    ].to_dict(orient="records")
    (out / "ultimate_selected_combos.json").write_text(
        json.dumps(selected_combos, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    config_rows = []
    for combo in selected_combos:
        cols, meta = select_feature_set(frame, str(combo["variant"]))
        model_name = str(combo["model_name"])
        anchor = "logistic" if model_name != "logistic" else "logistic_c0_15"
        for train_name, max_train in TRAINING_WINDOWS:
            for calibration in CALIBRATION_MODES:
                for routing in ROUTING_MODES:
                    wf, _ = run_walk_forward(
                        frame,
                        cols,
                        min_train=min_train,
                        oos_block=oos_block,
                        random_state=random_state,
                        candidate_names=[model_name, anchor],
                        calibration_mode=calibration,
                        routing_mode=routing,
                        prediction_mode=f"fixed:{model_name}",
                        max_train_rows=max_train,
                    )
                    summary = summarize_oos(wf)
                    config_rows.append({
                        **meta,
                        "model_name": model_name,
                        "training_window": train_name,
                        "max_train_rows": max_train,
                        "calibration_mode": calibration,
                        "routing_mode": routing,
                        **summary,
                    })

    config_table = pd.DataFrame(config_rows)
    config_table.to_csv(out / "ultimate_configuration_screen.csv", index=False)

    baseline_mask = (
        (config_table["training_window"] == "expanding")
        & (config_table["calibration_mode"] == "full")
        & (config_table["routing_mode"] == "dynamic")
    )
    baselines = config_table.loc[baseline_mask].copy()
    if baselines.empty:
        raise RuntimeError("Ultimate configuration baseline rows are missing")

    winners = (
        config_table.sort_values(
            ["confirm_logloss", "confirm_brier", "confirm_ece",
             "training_window", "calibration_mode", "routing_mode"],
            kind="mergesort",
        )
        .groupby(["variant", "model_name"], as_index=False, sort=False)
        .first()
    )
    winner = winners.sort_values(
        ["confirm_logloss", "confirm_brier", "confirm_ece", "variant", "model_name"],
        kind="mergesort",
    ).iloc[0].to_dict()

    winner_baseline = baselines.loc[
        (baselines["variant"] == winner["variant"])
        & (baselines["model_name"] == winner["model_name"])
    ].sort_values(["confirm_logloss", "confirm_brier"], kind="mergesort").iloc[0]

    summary = {
        "status": "RESEARCH_EXECUTED",
        "feature_variants_tested": len(features),
        "base_models": list(BASE_MODELS),
        "model_variants_tested": len(MODEL_ECOLOGY),
        "configuration_rows": int(len(config_table)),
        "locked_blocks": LOCKED_BLOCKS,
        "winner": winner,
        "winner_vs_same_combo_baseline": {
            "confirm_logloss_delta": float(winner["confirm_logloss"] - winner_baseline["confirm_logloss"]),
            "locked_logloss_delta": float(winner["locked_logloss"] - winner_baseline["locked_logloss"]),
            "confirm_brier_delta": float(winner["confirm_brier"] - winner_baseline["confirm_brier"]),
            "locked_brier_delta": float(winner["locked_brier"] - winner_baseline["locked_brier"]),
            "confirm_accuracy_delta": float(winner["confirm_accuracy"] - winner_baseline["confirm_accuracy"]),
            "locked_accuracy_delta": float(winner["locked_accuracy"] - winner_baseline["locked_accuracy"]),
            "confirm_ece_delta": float(winner["confirm_ece"] - winner_baseline["confirm_ece"]),
            "locked_ece_delta": float(winner["locked_ece"] - winner_baseline["locked_ece"]),
        },
        "selection_firewall": {
            "feature_selection": "screen_blocks_only",
            "model_selection": "confirm_blocks_only_after_feature_screen",
            "configuration_selection": "confirm_blocks_only",
            "locked_oos_used_for_selection": False,
            "frozen_holdout_used_for_selection": False,
            "same_oos_window_required": True,
        },
        "safety_contract": {
            "pit_verified_input_required": True,
            "production_changed": False,
            "production_registry_changed": False,
        },
    }
    (out / "ultimate_matrix_status.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the staged ultimate soccer research matrix.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-train", type=int, default=2000)
    parser.add_argument("--oos-block", type=int, default=4000)
    parser.add_argument("--top-features", type=int, default=6)
    parser.add_argument("--top-combos", type=int, default=3)
    args = parser.parse_args()

    result = run_ultimate_matrix(
        args.input,
        args.output_dir,
        min_train=args.min_train,
        oos_block=args.oos_block,
        top_features=args.top_features,
        top_combos=args.top_combos,
    )
    print(json.dumps(result, ensure_ascii=False, default=str))
