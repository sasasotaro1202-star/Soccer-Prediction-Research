"""Generate research-only candidates from matured soccer prediction experience.

The candidate planner is intentionally downstream of settlement. It consumes only
PIT-valid, outcome-matured experience and proposes what should be investigated next.
It never changes production predictions, thresholds, models, or holdout policy.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import classification_metrics

SCHEMA_VERSION = 1
MAX_CANDIDATES = 20
MIN_OBSERVATIONS = 30
MIN_LOGLOSS_IMPACT = 0.02
MIN_ACCURACY_GAP = 0.03
LABELS = ("H", "D", "A")


def _candidate_id(dimension: str, segment: str) -> str:
    raw = f"{dimension}|{segment}"
    return f"exp-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]}"


def _score(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {"n": 0}
    p = frame[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    y = frame["actual_result"].map({"H": 0, "D": 1, "A": 2}).to_numpy(dtype=int)
    metrics = classification_metrics(y, p)
    return {
        "n": int(len(frame)),
        "logloss": float(metrics["logloss"]),
        "brier": float(metrics["brier"]),
        "ece": float(metrics["ece"]),
        "accuracy": float((p.argmax(axis=1) == y).mean()),
    }


def _confidence_bucket(confidence: pd.Series) -> pd.Series:
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0000001]
    labels = ["0.0-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    return pd.cut(
        pd.to_numeric(confidence, errors="coerce"),
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    ).astype("string")


def _validated_settled(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "match_id",
        "kickoff_utc",
        "prediction_pit_cutoff_utc",
        "prediction_pit_gate",
        "experience_available_at_utc",
        "p_home",
        "p_draw",
        "p_away",
        "actual_result",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"experience candidate input missing required columns: {missing}")

    d = frame.copy()
    d["kickoff_utc"] = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    d["prediction_pit_cutoff_utc"] = pd.to_datetime(
        d["prediction_pit_cutoff_utc"], utc=True, errors="coerce"
    )
    d["experience_available_at_utc"] = pd.to_datetime(
        d["experience_available_at_utc"], utc=True, errors="coerce"
    )
    d["actual_result"] = d["actual_result"].astype("string").str.strip().str.upper()
    for c in ("p_home", "p_draw", "p_away"):
        d[c] = pd.to_numeric(d[c], errors="coerce")

    valid = d["match_id"].notna() & d["kickoff_utc"].notna()
    valid &= d["prediction_pit_gate"].astype("string").eq("PASS")
    valid &= d["prediction_pit_cutoff_utc"].notna()
    valid &= d["experience_available_at_utc"].notna()
    valid &= d["prediction_pit_cutoff_utc"] < d["kickoff_utc"]
    valid &= d["experience_available_at_utc"] > d["kickoff_utc"]
    valid &= d["experience_available_at_utc"] > d["prediction_pit_cutoff_utc"]
    valid &= d["actual_result"].isin(LABELS)
    p = d[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    valid &= np.isfinite(p).all(axis=1)
    valid &= (p >= 0).all(axis=1)
    valid &= np.sum(p, axis=1) > 0

    d = d.loc[valid].copy()
    p = d[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    p /= p.sum(axis=1, keepdims=True)
    d[["p_home", "p_draw", "p_away"]] = p
    d["confidence"] = d[["p_home", "p_draw", "p_away"]].max(axis=1)
    d["confidence_bucket"] = _confidence_bucket(d["confidence"])

    if "prediction_state_id" in d.columns:
        ids = d["prediction_state_id"].astype("string").str.strip()
        if ids.isna().any() or ids.eq("").any():
            raise RuntimeError("experience candidate input contains missing prediction_state_id")
        if ids.duplicated().any():
            raise RuntimeError("experience candidate input contains duplicate prediction_state_id")
    elif d["match_id"].duplicated().any():
        raise RuntimeError(
            "experience candidate input contains duplicate match_id without prediction_state_id"
        )

    return d.reset_index(drop=True)


def _candidate_row(
    dimension: str,
    segment: str,
    frame: pd.DataFrame,
    global_score: dict[str, float | int],
    action: str,
    hypothesis: str,
) -> dict[str, Any] | None:
    if len(frame) < MIN_OBSERVATIONS:
        return None
    score = _score(frame)
    logloss_impact = float(score["logloss"]) - float(global_score["logloss"])
    accuracy_gap = float(global_score["accuracy"]) - float(score["accuracy"])
    if logloss_impact < MIN_LOGLOSS_IMPACT and accuracy_gap < MIN_ACCURACY_GAP:
        return None
    return {
        "candidate_id": _candidate_id(dimension, segment),
        "source_dimension": dimension,
        "source_segment": segment,
        "evidence_n": int(score["n"]),
        "evidence_logloss": round(float(score["logloss"]), 6),
        "evidence_brier": round(float(score["brier"]), 6),
        "evidence_ece": round(float(score["ece"]), 6),
        "evidence_accuracy": round(float(score["accuracy"]), 6),
        "global_logloss": round(float(global_score["logloss"]), 6),
        "global_accuracy": round(float(global_score["accuracy"]), 6),
        "impact_vs_global_logloss": round(logloss_impact, 6),
        "accuracy_gap_vs_global": round(accuracy_gap, 6),
        "action": action,
        "hypothesis": hypothesis,
        "research_only": True,
        "production_changed": False,
        "frozen_holdout_allowed": False,
        "promotion_status": "HOLD_UNTIL_CHRONOLOGICAL_OOS",
    }


def build_experience_candidate_plan(
    frame: pd.DataFrame,
    *,
    max_candidates: int = MAX_CANDIDATES,
) -> dict[str, Any]:
    data = _validated_settled(frame)
    if data.empty:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "WARMUP",
            "source": "matured_experience_ledger",
            "settled_rows": 0,
            "candidates": [],
            "safety_contract": {
                "research_only": True,
                "production_changed": False,
                "frozen_holdout_allowed": False,
                "promotion_requires_chronological_oos": True,
                "outcome_data_used_only_after_maturity": True,
            },
        }

    global_score = _score(data)
    proposals: list[dict[str, Any]] = []

    for dimension, grouped in (
        ("competition", data.groupby("competition", dropna=False)),
        ("model", data.groupby("model_version", dropna=False)),
        ("confidence_bucket", data.groupby("confidence_bucket", dropna=False)),
    ):
        for segment, frame_part in grouped:
            if pd.isna(segment) or str(segment).strip() == "":
                continue
            candidate = _candidate_row(
                dimension,
                str(segment),
                frame_part,
                global_score,
                action="segment_specific_model_or_calibration_challenge",
                hypothesis=(
                    f"Test whether {dimension}={segment} has a persistent error structure "
                    "that justifies a dedicated model, calibration, route, or information policy."
                ),
            )
            if candidate is not None:
                proposals.append(candidate)

    predicted_idx = data[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float).argmax(axis=1)
    actual_idx = data["actual_result"].map({"H": 0, "D": 1, "A": 2}).to_numpy(dtype=int)
    correct = pd.Series(predicted_idx == actual_idx, index=data.index)
    error_specs = [
        (
            "high_confidence_wrong",
            (data["confidence"] >= 0.75) & ~correct,
            "high_confidence_abstention_challenge",
            "Test whether high-confidence wrong cases are predictable from outcome-free telemetry and can support selective prediction without degrading normal cases.",
        ),
        (
            "high_model_disagreement",
            pd.to_numeric(data.get("shadow_model_disagreement"), errors="coerce") >= 0.20,
            "disagreement_trigger_challenge",
            "Test whether high model disagreement predicts elevated future error and supports routing, abstention, or fallback.",
        ),
        (
            "high_uncertainty",
            pd.to_numeric(data.get("shadow_uncertainty_score"), errors="coerce") >= 0.60,
            "uncertainty_policy_challenge",
            "Test whether outcome-free uncertainty telemetry identifies a recurring high-loss regime suitable for selective prediction or recalibration.",
        ),
        (
            "high_routing_risk",
            pd.to_numeric(data.get("shadow_routing_risk"), errors="coerce") >= 0.50,
            "routing_risk_challenge",
            "Test whether routing-risk telemetry identifies cases where the dynamic route needs an OOS-validated fallback or alternative information path.",
        ),
    ]

    for segment, mask, action, hypothesis in error_specs:
        if hasattr(mask, "fillna"):
            mask = mask.fillna(False)
        candidate = _candidate_row(
            "error_types",
            segment,
            data.loc[mask],
            global_score,
            action=action,
            hypothesis=hypothesis,
        )
        if candidate is not None:
            proposals.append(candidate)

    proposals.sort(
        key=lambda row: (
            -float(row["impact_vs_global_logloss"]),
            -int(row["evidence_n"]),
            row["candidate_id"],
        )
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "READY" if proposals else "WARMUP",
        "source": "matured_experience_ledger",
        "settled_rows": int(len(data)),
        "candidates": proposals[: int(max_candidates)],
        "safety_contract": {
            "research_only": True,
            "production_changed": False,
            "frozen_holdout_allowed": False,
            "promotion_requires_chronological_oos": True,
            "outcome_data_used_only_after_maturity": True,
        },
    }
