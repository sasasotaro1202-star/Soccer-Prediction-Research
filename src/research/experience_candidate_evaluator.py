"""Research-only evaluation of experience-derived candidates on scoped OOS evidence.

The candidate plan is generated from matured outcomes, while the OOS case artifact is
an independent chronological evaluation artifact. Because the two artifacts can
overlap historically, every result here is explicitly diagnostic reuse only:
promotion requires a fresh chronological OOS period created strictly after the
candidate hypothesis became known.

No production model or holdout artifact is modified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
LOCKED_BLOCKS = 2
MIN_FOLDS = 3
MIN_ROWS_PER_FOLD = 30
MIN_RELATIVE_IMPROVEMENT = 0.03
MIN_ABSOLUTE_LOGLOSS_IMPACT = 0.02
MIN_ABSOLUTE_ACCURACY_GAP = 0.03


def _score(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {"n": 0, "logloss": np.nan, "accuracy": np.nan, "brier": np.nan, "ece": np.nan}
    y = pd.to_numeric(frame["actual"], errors="coerce").to_numpy(dtype=int)
    p = frame[["p_home", "p_draw", "p_away"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Scoped OOS probabilities are invalid")
    logloss = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1.0)).mean())
    accuracy = float((p.argmax(axis=1) == y).mean())
    one_hot = np.eye(3, dtype=float)[y]
    brier = float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))
    confidences = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for i in range(10):
        mask = (confidences >= edges[i]) & (
            (confidences < edges[i + 1]) if i < 9 else (confidences <= edges[i + 1])
        )
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidences[mask].mean()))
    return {"n": int(len(y)), "logloss": logloss, "accuracy": accuracy, "brier": brier, "ece": float(ece)}


def _fingerprint(frame: pd.DataFrame) -> str:
    cols = [c for c in ("match_id", "kickoff_utc", "competition") if c in frame.columns]
    if not cols:
        raise ValueError("Cannot fingerprint OOS slice without match identity")
    rows = frame[cols].astype("string").fillna("").sort_values(cols, kind="mergesort")
    payload = rows.to_csv(index=False, lineterminator="
").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prepare(cases: pd.DataFrame, blocks: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "match_id", "oos_start", "competition", "kickoff_utc", "actual",
        "correct", "p_home", "p_draw", "p_away", "confidence",
        "model_disagreement", "uncertainty_score", "risk_score",
    }
    missing = sorted(required - set(cases.columns))
    if missing:
        raise ValueError(f"experience candidate scoped-OOS cases missing columns: {missing}")
    d = cases.copy()
    d["match_id"] = d["match_id"].astype("string").str.strip()
    d["competition"] = d["competition"].astype("string").str.strip().str.upper()
    d["oos_start"] = pd.to_datetime(d["oos_start"], utc=True, errors="coerce")
    d["kickoff_utc"] = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    for c in ("actual", "correct"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ("p_home", "p_draw", "p_away", "confidence", "model_disagreement", "uncertainty_score", "risk_score"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    valid = (
        d["match_id"].notna() & d["match_id"].ne("")
        & d["competition"].notna() & d["competition"].ne("")
        & d["oos_start"].notna() & d["kickoff_utc"].notna()
        & d["actual"].isin([0, 1, 2])
        & d["correct"].isin([0, 1])
        & d[["p_home", "p_draw", "p_away"]].notna().all(axis=1)
        & d[["confidence", "model_disagreement", "uncertainty_score", "risk_score"]].notna().all(axis=1)
    )
    if not bool(valid.all()):
        raise ValueError("experience candidate scoped-OOS contains invalid rows")
    p = d[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p.sum(axis=1) <= 0).any():
        raise ValueError("experience candidate scoped-OOS contains invalid probabilities")
    d[["p_home", "p_draw", "p_away"]] = p / p.sum(axis=1, keepdims=True)
    if d.duplicated(["match_id"], keep=False).any():
        raise ValueError("experience candidate scoped-OOS requires one case row per fixture")
    d["confidence_bucket"] = pd.cut(
        d["confidence"],
        bins=[0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.000001],
        labels=["0.0-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"],
        right=False,
        include_lowest=True,
    ).astype("string")
    block_rows = d.groupby("oos_start", sort=True).size().reset_index(name="n")
    if blocks is not None and not blocks.empty and {"oos_start", "oos_end", "n"}.issubset(blocks.columns):
        b = blocks.copy()
        b["oos_start"] = pd.to_datetime(b["oos_start"], utc=True, errors="coerce")
        b["oos_end"] = pd.to_datetime(b["oos_end"], utc=True, errors="coerce")
        b["n"] = pd.to_numeric(b["n"], errors="coerce")
        if b["oos_start"].isna().any() or b["oos_end"].isna().any():
            raise ValueError("experience candidate scoped-OOS block boundaries are invalid")
        if b["oos_start"].duplicated().any():
            raise ValueError("experience candidate scoped-OOS has duplicate block starts")
        b = b.sort_values("oos_start", kind="mergesort").reset_index(drop=True)
    else:
        starts = sorted(d["oos_start"].drop_duplicates())
        b = pd.DataFrame({"oos_start": starts})
        b["oos_end"] = [
            d.loc[d["oos_start"].eq(start), "kickoff_utc"].max() for start in starts
        ]
        b["n"] = [
            int(d["oos_start"].eq(start).sum()) for start in starts
        ]
    if len(b) < LOCKED_BLOCKS + 1:
        raise ValueError("experience candidate scoped-OOS requires development plus two locked blocks")
    locked_starts = set(b.tail(LOCKED_BLOCKS)["oos_start"].tolist())
    dev = d[~d["oos_start"].isin(locked_starts)].copy()
    return d, b


def _scope_mask(d: pd.DataFrame, dimension: str, segment: str) -> pd.Series:
    if dimension == "competition":
        return d["competition"].eq(segment.upper())
    if dimension == "confidence_bucket":
        return d["confidence_bucket"].eq(segment)
    if dimension == "error_types":
        if segment == "high_confidence_wrong":
            return (d["confidence"] >= 0.75) & d["correct"].eq(0)
        if segment == "high_model_disagreement":
            return d["model_disagreement"] >= 0.20
        if segment == "high_uncertainty":
            return d["uncertainty_score"] >= 0.60
        if segment == "high_routing_risk":
            return d["risk_score"] >= 0.50
    return pd.Series(False, index=d.index)


def _fold_stats(frame: pd.DataFrame, blocks: pd.DataFrame) -> tuple[list[dict[str, Any]], int]:
    rows = []
    for start in blocks["oos_start"].tolist():
        part = frame[frame["oos_start"].eq(start)]
        if part.empty:
            continue
        metric = _score(part)
        rows.append({
            "oos_start": start.isoformat(),
            "n": int(metric["n"]),
            "logloss": float(metric["logloss"]),
            "accuracy": float(metric["accuracy"]),
            "brier": float(metric["brier"]),
            "ece": float(metric["ece"]),
            "fingerprint": _fingerprint(part),
        })
    return rows, min((int(row["n"]) for row in rows), default=0)


def evaluate_experience_candidates(
    candidate_plan: dict[str, Any],
    cases: pd.DataFrame,
    blocks: pd.DataFrame | None = None,
    *,
    min_folds: int = MIN_FOLDS,
    min_rows_per_fold: int = MIN_ROWS_PER_FOLD,
    min_relative_improvement: float = MIN_RELATIVE_IMPROVEMENT,
) -> dict[str, Any]:
    safety = candidate_plan.get("safety_contract")
    if not isinstance(safety, dict):
        raise ValueError("candidate plan safety contract missing")
    if safety.get("research_only") is not True:
        raise ValueError("candidate plan must remain research_only")
    if safety.get("production_changed") is not False:
        raise ValueError("candidate plan claims production mutation")
    if safety.get("frozen_holdout_allowed") is not False:
        raise ValueError("candidate plan cannot use frozen holdout")
    if min_folds < 3:
        raise ValueError("experience candidate evaluation requires >=3 OOS folds")
    if min_rows_per_fold < 10:
        raise ValueError("experience candidate evaluation requires >=10 rows/fold")
    if not 0.0 <= min_relative_improvement <= 1.0:
        raise ValueError("invalid minimum relative improvement")

    d, block_frame = _prepare(cases, blocks)
    locked_starts = set(block_frame.tail(LOCKED_BLOCKS)["oos_start"].tolist())
    dev = d[~d["oos_start"].isin(locked_starts)].copy()
    global_metric = _score(dev)
    global_fingerprint = _fingerprint(dev)
    results: list[dict[str, Any]] = []

    for candidate in candidate_plan.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        dimension = str(candidate.get("source_dimension") or "").strip()
        segment = str(candidate.get("source_segment") or "").strip()
        base = {
            "candidate_id": str(candidate.get("candidate_id") or ""),
            "source_dimension": dimension,
            "source_segment": segment,
            "experience_evidence_n": int(candidate.get("evidence_n") or 0),
            "experience_signal_logloss_impact": float(candidate.get("impact_vs_global_logloss") or 0.0),
            "research_only": True,
            "promotion_allowed": False,
            "fresh_oos_required": True,
            "locked_holdout_used": False,
            "evaluation_mode": "DIAGNOSTIC_REUSE_OF_EXISTING_OOS",
        }
        if dimension == "model":
            base.update({
                "status": "NO_DIRECT_OOS_SCOPE",
                "reason": "oos_case_diagnostics does not preserve prediction model_version at fixture level",
            })
            results.append(base)
            continue
        mask = _scope_mask(dev, dimension, segment)
        scoped = dev.loc[mask].copy()
        if scoped.empty:
            base.update({
                "status": "NO_SCOPE_ROWS",
                "reason": "candidate segment has no rows in development OOS",
            })
            results.append(base)
            continue
        folds = sorted(scoped["oos_start"].drop_duplicates().tolist())
        block_stats, min_rows = _fold_stats(scoped, block_frame.loc[~block_frame["oos_start"].isin(locked_starts)])
        metric = _score(scoped)
        gap_logloss = float(metric["logloss"] - global_metric["logloss"])
        gap_accuracy = float(global_metric["accuracy"] - metric["accuracy"])
        eligible = len(folds) >= int(min_folds) and min_rows >= int(min_rows_per_fold)
        relative_degradation = gap_logloss / max(abs(float(global_metric["logloss"])), 1e-9)
        diagnostic_signal = bool(
            eligible and (
                gap_logloss >= MIN_ABSOLUTE_LOGLOSS_IMPACT
                or gap_accuracy >= MIN_ABSOLUTE_ACCURACY_GAP
            )
        )
        base.update({
            "status": "EVALUATED" if eligible else "INSUFFICIENT_OOS_EVIDENCE",
            "folds": int(len(folds)),
            "min_rows_per_fold": int(min_rows),
            "global_fingerprint": global_fingerprint,
            "scoped_fingerprint": _fingerprint(scoped),
            "global_development": global_metric,
            "scoped_development": metric,
            "logloss_delta_scoped_minus_global": gap_logloss,
            "accuracy_delta_scoped_minus_global": float(metric["accuracy"] - global_metric["accuracy"]),
            "brier_delta_scoped_minus_global": float(metric["brier"] - global_metric["brier"]),
            "ece_delta_scoped_minus_global": float(metric["ece"] - global_metric["ece"]),
            "relative_logloss_degradation": relative_degradation,
            "candidate_passes_diagnostic_screen": diagnostic_signal,
            "folds_detail": block_stats,
            "next_action": (
                "CREATE_FRESH_CHRONOLOGICAL_OOS_CANDIDATE"
                if diagnostic_signal
                else "HOLD"
            ),
        })
        results.append(base)

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "EVALUATED" if results else "NO_CANDIDATES",
        "candidate_count": len(results),
        "evaluations": results,
        "policy": {
            "research_only": True,
            "promotion_allowed": False,
            "fresh_oos_required": True,
            "locked_holdout_used": False,
            "diagnostic_reuse_only": True,
            "min_folds": int(min_folds),
            "min_rows_per_fold": int(min_rows_per_fold),
            "min_relative_improvement": float(min_relative_improvement),
            "min_absolute_logloss_impact": MIN_ABSOLUTE_LOGLOSS_IMPACT,
            "min_absolute_accuracy_gap": MIN_ABSOLUTE_ACCURACY_GAP,
            "locked_blocks_excluded_from_candidate_diagnostic": int(LOCKED_BLOCKS),
        },
    }


def write_evaluation(
    candidate_path: str = "artifacts/experience_candidate_plan.json",
    cases_path: str = "artifacts/oos_case_diagnostics.csv",
    blocks_path: str = "artifacts/oos_metrics.csv",
    output_path: str = "artifacts/experience_candidate_evaluation.json",
) -> dict[str, Any]:
    candidate_file = Path(candidate_path)
    cases_file = Path(cases_path)
    blocks_file = Path(blocks_path)
    if not candidate_file.is_file() or candidate_file.stat().st_size == 0:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "DEFERRED_NO_CANDIDATE_PLAN",
            "candidate_count": 0,
            "evaluations": [],
            "policy": {
                "research_only": True,
                "promotion_allowed": False,
                "fresh_oos_required": True,
                "locked_holdout_used": False,
            },
        }
    if not cases_file.is_file() or cases_file.stat().st_size == 0:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "DEFERRED_NO_OOS_CASES",
            "candidate_count": 0,
            "evaluations": [],
            "policy": {
                "research_only": True,
                "promotion_allowed": False,
                "fresh_oos_required": True,
                "locked_holdout_used": False,
            },
        }
    plan = json.loads(candidate_file.read_text(encoding="utf-8"))
    cases = pd.read_csv(cases_file)
    blocks = pd.read_csv(blocks_file) if blocks_file.is_file() and blocks_file.stat().st_size else None
    result = evaluate_experience_candidates(plan, cases, blocks)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return result
