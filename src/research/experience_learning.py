"""PIT-safe learning from matured prediction experience.

This module learns a research-only empirical correction policy from settled
prediction history. Each target prediction is allowed to use only outcomes whose
teacher information was already available at that target's prediction cutoff.

The learned policy is deliberately a Challenger artifact: it never changes the
production model by itself. Promotion requires chronological OOS evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import classification_metrics

LEDGER = Path("data/experience/prediction_ledger.csv")
POLICY = Path("artifacts/experience_policy_candidate.json")
METRICS = Path("artifacts/experience_learning_oos.csv")
STATUS = Path("artifacts/experience_learning_status.json")

LABELS = ("H", "D", "A")
CONFIDENCE_BINS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0000001)
ALPHAS = (5.0, 10.0, 20.0, 40.0, 80.0)
MIN_TEACHER_ROWS = 5
MIN_ROWS_PER_BLOCK = 30
MIN_BLOCKS = 3


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _now() -> str:
    return pd.Timestamp.utcnow().isoformat()


def _confidence_bucket(value: float) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(v):
        return "unknown"
    for left, right in zip(CONFIDENCE_BINS[:-1], CONFIDENCE_BINS[1:]):
        if left <= v < right:
            return f"{left:.1f}-{min(right, 1.0):.1f}"
    return "unknown"


def _predicted_class(row: pd.Series) -> str:
    p = np.asarray([row["p_home"], row["p_draw"], row["p_away"]], dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        return ""
    return LABELS[int(np.argmax(p))]


def _validate_ledger(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "match_id",
        "kickoff_utc",
        "prediction_pit_cutoff_utc",
        "prediction_pit_gate",
        "experience_available_at_utc",
        "competition",
        "model_version",
        "p_home",
        "p_draw",
        "p_away",
        "actual_result",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"experience learning ledger missing required columns: {missing}")

    d = frame.copy()
    d["kickoff_utc"] = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    d["prediction_pit_cutoff_utc"] = pd.to_datetime(
        d["prediction_pit_cutoff_utc"], utc=True, errors="coerce"
    )
    d["experience_available_at_utc"] = pd.to_datetime(
        d["experience_available_at_utc"], utc=True, errors="coerce"
    )

    if d["match_id"].isna().any() or d["match_id"].astype(str).str.strip().eq("").any():
        raise RuntimeError("experience learning ledger contains missing match_id")
    if d["match_id"].duplicated().any():
        # A fixture can legitimately have multiple prediction states, so only
        # duplicate prediction-state rows are unsafe. When available, use it.
        if "prediction_state_id" in d.columns and d["prediction_state_id"].duplicated().any():
            raise RuntimeError("experience learning ledger contains duplicate prediction_state_id")

    for c in ("p_home", "p_draw", "p_away"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    valid_p = np.isfinite(d[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)).all(axis=1)
    valid_p &= (d[["p_home", "p_draw", "p_away"]] >= 0).all(axis=1)
    valid_p &= d[["p_home", "p_draw", "p_away"]].sum(axis=1).gt(0)
    d = d.loc[valid_p].copy()
    d[["p_home", "p_draw", "p_away"]] = (
        d[["p_home", "p_draw", "p_away"]]
        .div(d[["p_home", "p_draw", "p_away"]].sum(axis=1), axis=0)
    )

    pit_ok = d["prediction_pit_gate"].astype("string").eq("PASS")
    pit_ok &= d["prediction_pit_cutoff_utc"].notna()
    pit_ok &= d["kickoff_utc"].notna()
    pit_ok &= d["experience_available_at_utc"].notna()
    pit_ok &= d["prediction_pit_cutoff_utc"] < d["kickoff_utc"]
    pit_ok &= d["experience_available_at_utc"] > d["prediction_pit_cutoff_utc"]
    d = d.loc[pit_ok].copy()

    d["actual_result"] = d["actual_result"].astype("string").str.strip().str.upper()
    d = d.loc[d["actual_result"].isin(LABELS)].copy()
    d["confidence"] = d[["p_home", "p_draw", "p_away"]].max(axis=1)
    d["predicted_class"] = d.apply(_predicted_class, axis=1)
    d["confidence_bucket"] = d["confidence"].map(_confidence_bucket)
    d = d.sort_values(["prediction_pit_cutoff_utc", "kickoff_utc", "match_id"], kind="mergesort")
    return d.reset_index(drop=True)


def _teacher_pool(history: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    cutoff = target["prediction_pit_cutoff_utc"]
    if pd.isna(cutoff):
        return history.iloc[0:0]

    eligible = history[
        (history["experience_available_at_utc"] <= cutoff)
        & (history["prediction_pit_cutoff_utc"] < cutoff)
        & (history["actual_result"].isin(LABELS))
        & (history["experience_available_at_utc"] > history["prediction_pit_cutoff_utc"])
    ].copy()
    if eligible.empty:
        return eligible

    comp = str(target.get("competition", ""))
    model = str(target.get("model_version", ""))
    pred_class = str(target.get("predicted_class", ""))
    bucket = str(target.get("confidence_bucket", ""))
    predicates = [
        ("competition_model_class_confidence", (eligible["competition"].astype(str) == comp)
         & (eligible["model_version"].astype(str) == model)
         & (eligible["predicted_class"] == pred_class)
         & (eligible["confidence_bucket"] == bucket)),
        ("competition_class_confidence", (eligible["competition"].astype(str) == comp)
         & (eligible["predicted_class"] == pred_class)
         & (eligible["confidence_bucket"] == bucket)),
        ("competition_class", (eligible["competition"].astype(str) == comp)
         & (eligible["predicted_class"] == pred_class)),
        ("class_confidence", (eligible["predicted_class"] == pred_class)
         & (eligible["confidence_bucket"] == bucket)),
        ("class", eligible["predicted_class"] == pred_class),
        ("global", pd.Series(True, index=eligible.index)),
    ]
    for name, mask in predicates:
        pool = eligible.loc[mask].copy()
        if len(pool) >= MIN_TEACHER_ROWS:
            pool.attrs["source"] = name
            return pool
    eligible.attrs["source"] = "global_insufficient_history"
    return eligible


def _posterior(pool: pd.DataFrame, alpha: float) -> np.ndarray:
    counts = np.asarray(
        [int((pool["actual_result"] == label).sum()) for label in LABELS], dtype=float
    )
    return (counts + alpha / 3.0) / (counts.sum() + alpha)


def _experience_adjusted_probability(
    baseline: np.ndarray, pool: pd.DataFrame, alpha: float
) -> tuple[np.ndarray, float, str, int]:
    if pool.empty:
        return baseline.copy(), 0.0, "none", 0
    posterior = _posterior(pool, alpha)
    n = int(len(pool))
    weight = float(n / (n + alpha))
    adjusted = (1.0 - weight) * baseline + weight * posterior
    adjusted = np.clip(adjusted, 1e-6, 1.0)
    adjusted /= adjusted.sum()
    return adjusted, weight, str(pool.attrs.get("source", "unknown")), n


def _score_rows(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    if len(y) == 0:
        return {"n": 0}
    metrics = classification_metrics(y, p)
    return {
        "n": int(len(y)),
        "accuracy_pct": round(float((p.argmax(axis=1) == y).mean() * 100.0), 6),
        "logloss": round(float(metrics["logloss"]), 6),
        "brier": round(float(metrics["brier"]), 6),
        "rps": round(float(metrics["rps"]), 6),
        "ece": round(float(metrics["ece"]), 6),
    }


def _build_replay(
    data: pd.DataFrame, alpha: float, *, store_rows: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_records: list[dict[str, Any]] = []
    history = data.iloc[0:0].copy()
    for _, target in data.iterrows():
        base = target[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
        pool = _teacher_pool(history, target)
        adjusted, weight, source, n_teachers = _experience_adjusted_probability(
            base, pool, alpha
        )
        actual = LABELS.index(str(target["actual_result"]))
        baseline_records.append(
            {
                "match_id": str(target["match_id"]),
                "prediction_time_utc": target["prediction_pit_cutoff_utc"].isoformat(),
                "kickoff_utc": target["kickoff_utc"].isoformat(),
                "competition": str(target["competition"]),
                "model_version": str(target["model_version"]),
                "predicted_class": str(target["predicted_class"]),
                "confidence_bucket": str(target["confidence_bucket"]),
                "actual_result": str(target["actual_result"]),
                "baseline_p_home": float(base[0]),
                "baseline_p_draw": float(base[1]),
                "baseline_p_away": float(base[2]),
                "experience_p_home": float(adjusted[0]),
                "experience_p_draw": float(adjusted[1]),
                "experience_p_away": float(adjusted[2]),
                "experience_weight": float(weight),
                "teacher_rows": int(n_teachers),
                "teacher_source": source,
                "_y": actual,
            }
        )
        history = pd.concat([history, target.to_frame().T], ignore_index=True)

    replay = pd.DataFrame(baseline_records)
    if replay.empty:
        return replay, replay
    replay["_y"] = pd.to_numeric(replay["_y"], errors="coerce").astype(int)
    block_ids = _chronological_blocks(replay)
    replay["block"] = block_ids
    return replay, replay.loc[:, [c for c in replay.columns if not c.startswith("_")]].copy()


def _chronological_blocks(replay: pd.DataFrame) -> np.ndarray:
    n = len(replay)
    if n < MIN_BLOCKS * MIN_ROWS_PER_BLOCK:
        return np.array([], dtype=int)
    blocks = min(5, n // MIN_ROWS_PER_BLOCK)
    # Equal-sized chronological blocks, each with at least MIN_ROWS_PER_BLOCK rows.
    return np.floor(np.arange(n) * blocks / n).astype(int).clip(0, blocks - 1)


def _evaluate_blocks(replay: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if replay.empty or replay["block"].nunique() < MIN_BLOCKS:
        return pd.DataFrame()
    for block, frame in replay.groupby("block", sort=True):
        y = frame["_y"].to_numpy(dtype=int)
        baseline = frame[["baseline_p_home", "baseline_p_draw", "baseline_p_away"]].to_numpy(dtype=float)
        experience = frame[["experience_p_home", "experience_p_draw", "experience_p_away"]].to_numpy(dtype=float)
        b = _score_rows(y, baseline)
        e = _score_rows(y, experience)
        rows.append(
            {
                "block": int(block),
                "prediction_time_start": str(frame["prediction_time_utc"].min()),
                "prediction_time_end": str(frame["prediction_time_utc"].max()),
                "n": int(len(frame)),
                "baseline_accuracy_pct": b["accuracy_pct"],
                "experience_accuracy_pct": e["accuracy_pct"],
                "delta_accuracy_pct": round(float(e["accuracy_pct"] - b["accuracy_pct"]), 6),
                "baseline_logloss": b["logloss"],
                "experience_logloss": e["logloss"],
                "delta_logloss": round(float(e["logloss"] - b["logloss"]), 6),
                "baseline_brier": b["brier"],
                "experience_brier": e["brier"],
                "delta_brier": round(float(e["brier"] - b["brier"]), 6),
                "baseline_ece": b["ece"],
                "experience_ece": e["ece"],
                "delta_ece": round(float(e["ece"] - b["ece"]), 6),
            }
        )
    return pd.DataFrame(rows)


def _candidate_status(blocks: pd.DataFrame, replay: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    if replay.empty or blocks.empty or len(blocks) < MIN_BLOCKS or len(replay) < MIN_BLOCKS * MIN_ROWS_PER_BLOCK:
        return "INSUFFICIENT_EXPERIENCE", {
            "status_reason": f"At least {MIN_BLOCKS} chronological OOS blocks with {MIN_ROWS_PER_BLOCK} rows each are required.",
            "blocks": int(len(blocks)),
            "rows": int(len(replay)),
            "min_rows_required": int(MIN_BLOCKS * MIN_ROWS_PER_BLOCK),
        }

    locked = blocks.tail(2)
    development = blocks.iloc[:-2]
    locked_non_regressions = int(
        ((locked["delta_logloss"] <= 0.0) & (locked["delta_brier"] <= 0.0)).sum()
    )
    development_improvement = bool(
        (development["delta_logloss"] < 0.0).any()
        or (development["delta_brier"] < 0.0).any()
    )
    locked_pass = locked_non_regressions == len(locked)
    candidate = bool(locked_pass and development_improvement)
    return ("PROMOTION_CANDIDATE" if candidate else "HOLD"), {
        "status_reason": "Candidate requires development improvement and non-regression in both locked blocks.",
        "blocks": int(len(blocks)),
        "rows": int(len(replay)),
        "development_blocks": int(len(development)),
        "locked_blocks": int(len(locked)),
        "locked_non_regressions": locked_non_regressions,
        "locked_required": int(len(locked)),
        "development_improvement": development_improvement,
        "locked_pass": locked_pass,
    }


def learn(ledger_path: str | Path = LEDGER) -> dict[str, Any]:
    data = _validate_ledger(_read(Path(ledger_path)))
    if data.empty:
        status = {
            "status": "INSUFFICIENT_EXPERIENCE",
            "reason": "No PIT-valid settled prediction experience is available.",
            "rows": 0,
            "generated_at_utc": _now(),
            "fail_closed": True,
        }
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
        pd.DataFrame().to_csv(METRICS, index=False)
        POLICY.parent.mkdir(parents=True, exist_ok=True)
        POLICY.write_text(json.dumps({
            "status": "INSUFFICIENT_EXPERIENCE",
            "policy_type": "empirical_experience_correction",
            "usable_for_production": False,
            "reason": status["reason"],
        }, indent=2), encoding="utf-8")
        return status

    best_alpha = None
    alpha_scores: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        replay, _ = _build_replay(data, alpha)
        if replay.empty:
            continue
        blocks = _evaluate_blocks(replay)
        if blocks.empty:
            continue
        lock = blocks.tail(2)
        locked_logloss = float(lock["experience_logloss"].mean())
        locked_brier = float(lock["experience_brier"].mean())
        alpha_scores.append(
            {
                "alpha": alpha,
                "blocks": int(len(blocks)),
                "locked_logloss": round(locked_logloss, 6),
                "locked_brier": round(locked_brier, 6),
            }
        )
    # Hyperparameter selection is itself restricted to the development history:
    # prefer the alpha with the best mean development LogLoss, then Brier.
    if alpha_scores and len(data) >= MIN_BLOCKS * MIN_ROWS_PER_BLOCK:
        scored = []
        for alpha in ALPHAS:
            replay, _ = _build_replay(data, alpha)
            blocks = _evaluate_blocks(replay)
            if len(blocks) < MIN_BLOCKS:
                continue
            development = blocks.iloc[:-2]
            if development.empty:
                continue
            scored.append(
                (
                    float(development["experience_logloss"].mean()),
                    float(development["experience_brier"].mean()),
                    float(alpha),
                )
            )
        if scored:
            best_alpha = min(scored)[2]
    if best_alpha is None:
        best_alpha = 20.0

    replay, clean_replay = _build_replay(data, float(best_alpha))
    blocks = _evaluate_blocks(replay)
    status, gate = _candidate_status(blocks, replay)

    aggregate_y = replay["_y"].to_numpy(dtype=int) if not replay.empty else np.array([], dtype=int)
    aggregate_baseline = replay[["baseline_p_home", "baseline_p_draw", "baseline_p_away"]].to_numpy(dtype=float) if not replay.empty else np.empty((0, 3))
    aggregate_experience = replay[["experience_p_home", "experience_p_draw", "experience_p_away"]].to_numpy(dtype=float) if not replay.empty else np.empty((0, 3))
    aggregate_baseline_metrics = _score_rows(aggregate_y, aggregate_baseline)
    aggregate_experience_metrics = _score_rows(aggregate_y, aggregate_experience)

    policy = {
        "status": status,
        "policy_type": "empirical_experience_correction",
        "version": f"experience-v1-alpha-{best_alpha:g}",
        "usable_for_production": False,
        "training_source": str(ledger_path),
        "temporal_rule": "teacher experience_available_at_utc <= target prediction_pit_cutoff_utc",
        "teacher_gate": "teacher outcome available strictly after teacher prediction cutoff",
        "hyperparameter_selection": "development-only chronological replay; locked blocks excluded",
        "alpha": float(best_alpha),
        "min_teacher_rows": MIN_TEACHER_ROWS,
        "hierarchy": [
            "competition+model+predicted_class+confidence_bucket",
            "competition+predicted_class+confidence_bucket",
            "competition+predicted_class",
            "predicted_class+confidence_bucket",
            "predicted_class",
            "global",
        ],
        "gate": gate,
        "aggregate_baseline_metrics": aggregate_baseline_metrics,
        "aggregate_experience_metrics": aggregate_experience_metrics,
        "generated_at_utc": _now(),
    }
    POLICY.parent.mkdir(parents=True, exist_ok=True)
    POLICY.write_text(json.dumps(policy, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    _evaluate_blocks(replay).to_csv(METRICS, index=False)
    clean_replay.to_csv(METRICS.with_name("experience_learning_replay.csv"), index=False)

    status_doc = {
        "status": status,
        "policy_file": str(POLICY),
        "metrics_file": str(METRICS),
        "replay_file": str(METRICS.with_name("experience_learning_replay.csv")),
        "rows": int(len(replay)),
        "blocks": int(len(blocks)),
        "alpha": float(best_alpha),
        "gate": gate,
        "aggregate_baseline_metrics": aggregate_baseline_metrics,
        "aggregate_experience_metrics": aggregate_experience_metrics,
        "alpha_trials": alpha_scores,
        "generated_at_utc": _now(),
        "fail_closed": True,
    }
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status_doc, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return status_doc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default=str(LEDGER))
    args = parser.parse_args()
    print(json.dumps(learn(args.ledger), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
