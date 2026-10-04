from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.evaluation.walk_forward import run_walk_forward
from src.research.feature_set_variants import _family, select_feature_set

LOCKED_BLOCKS = 2
PERTURB_FAMILIES = (
    "strength",
    "rest",
    "form",
    "basic_stats",
    "advanced_stats",
    "derived_difference",
    "h2h",
    "momentum",
    "interaction",
)


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
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
        raise RuntimeError(f"Robustness input missing required columns: {missing}")
    d = frame.copy()
    d["kickoff_utc"] = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    d["prediction_cutoff_at_utc"] = pd.to_datetime(
        d["prediction_cutoff_at_utc"], utc=True, errors="coerce"
    )
    d["feature_source_max_available_at_utc"] = pd.to_datetime(
        d["feature_source_max_available_at_utc"], utc=True, errors="coerce"
    )
    if d[["kickoff_utc", "prediction_cutoff_at_utc", "feature_source_max_available_at_utc"]].isna().any().any():
        raise RuntimeError("Robustness input contains invalid temporal fields")
    pit = d["pit_verified"].astype(str).str.strip().str.lower()
    if not pit.isin({"true", "false", "1", "0", "yes", "no"}).all():
        raise RuntimeError("Robustness input contains ambiguous pit_verified values")
    d["pit_verified"] = pit.isin({"true", "1", "yes"})
    cutoff_ok = d["feature_source_max_available_at_utc"] <= d["prediction_cutoff_at_utc"]
    cutoff_ok &= d["prediction_cutoff_at_utc"] <= d["kickoff_utc"]
    if not bool((d["pit_verified"] & cutoff_ok).all()):
        raise RuntimeError("Robustness PIT validation failed")
    return d.sort_values(["kickoff_utc", "match_id"], kind="mergesort").reset_index(drop=True)


def _locked_match_ids(frame: pd.DataFrame, *, min_train: int, oos_block: int) -> set[str]:
    d = frame.loc[frame["pit_verified"]].sort_values(
        ["kickoff_utc", "match_id"], kind="mergesort"
    ).reset_index(drop=True)
    if len(d) < min_train + oos_block * (LOCKED_BLOCKS + 1):
        raise RuntimeError("Not enough PIT rows to identify locked blocks")
    start = min_train
    blocks: list[set[str]] = []
    while start < len(d):
        end = min(start + oos_block, len(d))
        blocks.append(set(d.iloc[start:end]["match_id"].astype(str)))
        start = end
    if len(blocks) < LOCKED_BLOCKS:
        raise RuntimeError("Could not identify locked OOS blocks")
    locked: set[str] = set()
    for block in blocks[-LOCKED_BLOCKS:]:
        locked.update(block)
    return locked


def _metrics_tail(wf: pd.DataFrame, locked_blocks: int = LOCKED_BLOCKS) -> dict[str, float | int | None]:
    if wf.empty or len(wf) < locked_blocks:
        raise RuntimeError("Robustness WFO result does not contain locked blocks")
    locked = wf.tail(locked_blocks)
    def weighted(col: str) -> float | None:
        if col not in locked.columns:
            return None
        values = pd.to_numeric(locked[col], errors="coerce")
        weights = pd.to_numeric(locked["n"], errors="coerce")
        ok = values.notna() & weights.notna() & (weights > 0)
        if not bool(ok.any()):
            return None
        return float(np.average(values.loc[ok], weights=weights.loc[ok]))
    return {
        "locked_blocks": int(len(locked)),
        "locked_n": int(pd.to_numeric(locked["n"], errors="coerce").sum()),
        "locked_logloss": weighted("logloss"),
        "locked_brier": weighted("brier"),
        "locked_accuracy": weighted("accuracy"),
        "locked_ece": weighted("ece"),
        "oos_window_signature": str(wf["oos_window_signature"].iloc[0]),
    }


def _anchor(model_name: str) -> str:
    return "logistic" if model_name != "logistic" else "logistic_c0_15"


def _run_config(
    frame: pd.DataFrame,
    feature_cols: list[str],
    winner: dict,
    *,
    min_train: int,
    oos_block: int,
) -> pd.DataFrame:
    training_window = str(winner["training_window"])
    max_train = None if training_window == "expanding" else int(winner["max_train_rows"])
    model_name = str(winner["model_name"])
    wf, _ = run_walk_forward(
        frame,
        feature_cols,
        min_train=min_train,
        oos_block=oos_block,
        random_state=42,
        candidate_names=[model_name, _anchor(model_name)],
        calibration_mode=str(winner["calibration_mode"]),
        routing_mode=str(winner["routing_mode"]),
        prediction_mode=f"fixed:{model_name}",
        max_train_rows=max_train,
    )
    return wf


def run_robustness(
    input_path: str,
    winner_path: str,
    output_dir: str,
    *,
    min_train: int = 2000,
    oos_block: int = 4000,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = _validate_frame(pd.read_csv(input_path))
    winner = json.loads(Path(winner_path).read_text(encoding="utf-8"))

    variant = str(winner["variant"])
    feature_cols, meta = select_feature_set(frame, variant)
    baseline_wf = _run_config(
        frame, feature_cols, winner, min_train=min_train, oos_block=oos_block
    )
    baseline = _metrics_tail(baseline_wf)

    rows: list[dict] = [{
        "perturbation": "baseline",
        "family": "",
        "feature_count": len(feature_cols),
        **baseline,
    }]

    family_groups: dict[str, list[str]] = {}
    for col in feature_cols:
        family_groups.setdefault(_family(col), []).append(col)

    for family in PERTURB_FAMILIES:
        drop = set(family_groups.get(family, []))
        if not drop:
            continue
        remaining = [c for c in feature_cols if c not in drop]
        if not remaining:
            continue
        wf = _run_config(frame, remaining, winner, min_train=min_train, oos_block=oos_block)
        m = _metrics_tail(wf)
        rows.append({
            "perturbation": "delete_feature_family",
            "family": family,
            "feature_count": len(remaining),
            **m,
            "delta_locked_logloss": (
                m["locked_logloss"] - baseline["locked_logloss"]
                if m["locked_logloss"] is not None and baseline["locked_logloss"] is not None
                else None
            ),
            "delta_locked_brier": (
                m["locked_brier"] - baseline["locked_brier"]
                if m["locked_brier"] is not None and baseline["locked_brier"] is not None
                else None
            ),
            "delta_locked_accuracy": (
                m["locked_accuracy"] - baseline["locked_accuracy"]
                if m["locked_accuracy"] is not None and baseline["locked_accuracy"] is not None
                else None
            ),
            "delta_locked_ece": (
                m["locked_ece"] - baseline["locked_ece"]
                if m["locked_ece"] is not None and baseline["locked_ece"] is not None
                else None
            ),
        })

    locked_ids = _locked_match_ids(frame, min_train=min_train, oos_block=oos_block)
    for family in ("form", "basic_stats", "advanced_stats"):
        family_cols = [c for c in family_groups.get(family, []) if c in feature_cols]
        if not family_cols:
            continue
        perturbed = frame.copy()
        locked_mask = perturbed["match_id"].astype(str).isin(locked_ids)
        perturbed.loc[locked_mask, family_cols] = np.nan
        wf = _run_config(
            perturbed,
            feature_cols,
            winner,
            min_train=min_train,
            oos_block=oos_block,
        )
        m = _metrics_tail(wf)
        rows.append({
            "perturbation": "locked_block_missingness",
            "family": family,
            "feature_count": len(feature_cols),
            **m,
            "delta_locked_logloss": (
                m["locked_logloss"] - baseline["locked_logloss"]
                if m["locked_logloss"] is not None and baseline["locked_logloss"] is not None
                else None
            ),
            "delta_locked_brier": (
                m["locked_brier"] - baseline["locked_brier"]
                if m["locked_brier"] is not None and baseline["locked_brier"] is not None
                else None
            ),
            "delta_locked_accuracy": (
                m["locked_accuracy"] - baseline["locked_accuracy"]
                if m["locked_accuracy"] is not None and baseline["locked_accuracy"] is not None
                else None
            ),
            "delta_locked_ece": (
                m["locked_ece"] - baseline["locked_ece"]
                if m["locked_ece"] is not None and baseline["locked_ece"] is not None
                else None
            ),
        })

    robustness = pd.DataFrame(rows)
    robustness.to_csv(out / "ultimate_robustness.csv", index=False)

    case_path = out / "winner_case_diagnostics.csv"
    case_wf, _ = run_walk_forward(
        frame,
        feature_cols,
        min_train=min_train,
        oos_block=oos_block,
        random_state=42,
        candidate_names=[str(winner["model_name"]), _anchor(str(winner["model_name"]))],
        calibration_mode=str(winner["calibration_mode"]),
        routing_mode=str(winner["routing_mode"]),
        prediction_mode=f"fixed:{winner['model_name']}",
        max_train_rows=(
            None if str(winner["training_window"]) == "expanding"
            else int(winner["max_train_rows"])
        ),
        case_output_path=str(case_path),
    )
    cases = pd.read_csv(case_path)
    case_rows = []
    for bucket, group in cases.groupby("risk_bucket", dropna=False):
        case_rows.append({
            "slice": "risk_bucket",
            "value": str(bucket),
            "n": int(len(group)),
            "accuracy": float(group["correct"].mean()),
            "mean_confidence": float(group["confidence"].mean()),
            "mean_risk_score": float(group["risk_score"].mean()),
        })
    for label, mask in {
        "confidence_ge_0.65": cases["confidence"] >= 0.65,
        "confidence_ge_0.75": cases["confidence"] >= 0.75,
        "high_risk": cases["risk_bucket"] == "HIGH",
        "low_support_risk": cases["history_support_risk"] >= 0.66,
        "high_model_disagreement": cases["model_disagreement"] >= 0.66,
    }.items():
        subset = cases.loc[mask]
        if subset.empty:
            continue
        case_rows.append({
            "slice": "case_condition",
            "value": label,
            "n": int(len(subset)),
            "accuracy": float(subset["correct"].mean()),
            "mean_confidence": float(subset["confidence"].mean()),
            "mean_risk_score": float(subset["risk_score"].mean()),
        })
    pd.DataFrame(case_rows).to_csv(out / "winner_case_slices.csv", index=False)

    status = {
        "status": "RESEARCH_EXECUTED",
        "winner": winner,
        "winner_feature_count": int(meta["feature_count"]),
        "robustness_rows": int(len(robustness)),
        "case_rows": int(len(cases)),
        "locked_blocks": LOCKED_BLOCKS,
        "production_changed": False,
        "production_registry_changed": False,
        "frozen_holdout_used_for_selection": False,
        "locked_oos_used_for_selection": False,
        "selection_authority": "upstream_ultimate_matrix_only",
        "robustness_is_diagnostic_not_selection": True,
    }
    (out / "ultimate_robustness_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return status


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run final-winner soccer robustness research.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--winner", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-train", type=int, default=2000)
    parser.add_argument("--oos-block", type=int, default=4000)
    args = parser.parse_args()
    result = run_robustness(
        args.input,
        args.winner,
        args.output_dir,
        min_train=args.min_train,
        oos_block=args.oos_block,
    )
    print(json.dumps(result, ensure_ascii=False, default=str))
