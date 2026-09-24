"""Chronological walk-forward evaluation for PIT-safe MOM ranking candidates.

Research-only: this module never creates or treats missing historical player data as
zero performance. It requires explicit PIT verification and exactly one MOM label
per match. Locked holdout blocks are evaluated only after all development choices
are fixed by the caller.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.models.mom_model import MOM_FEATURE_COLUMNS, fit_mom_model, predict_mom_distribution


@dataclass(frozen=True)
class MOMBlockMetrics:
    block: int
    train_matches: int
    test_matches: int
    test_rows: int
    top1_hit_rate: float
    top4_hit_rate: float
    mrr: float
    ndcg_at_4: float
    logloss: float
    brier: float
    ece: float
    mean_candidate_count: float
    min_candidate_count: int


def _validate_history(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "match_id",
        "player_id",
        "kickoff_utc",
        "feature_available_at_utc",
        "pit_verified",
        "is_motm",
        *MOM_FEATURE_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"MOM WFO data missing required columns: {missing}")

    d = frame.copy()
    d["kickoff_utc"] = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    d["feature_available_at_utc"] = pd.to_datetime(
        d["feature_available_at_utc"], utc=True, errors="coerce"
    )
    if d["kickoff_utc"].isna().any() or d["feature_available_at_utc"].isna().any():
        raise RuntimeError("MOM WFO timestamps are invalid or missing")
    d["pit_verified"] = d["pit_verified"].astype("boolean")
    if d["pit_verified"].isna().any():
        raise RuntimeError("MOM WFO pit_verified contains unknown values")
    d = d.loc[d["pit_verified"]].copy()
    if d.empty:
        raise RuntimeError("MOM WFO has no PIT-verified rows")

    d["match_id"] = d["match_id"].astype("string").str.strip()
    d["player_id"] = d["player_id"].astype("string").str.strip()
    if d["match_id"].eq("").any() or d["player_id"].eq("").any():
        raise RuntimeError("MOM WFO contains empty match_id/player_id")
    if d.duplicated(["match_id", "player_id"]).any():
        raise RuntimeError("MOM WFO contains duplicate match_id/player_id rows")
    if bool((d["feature_available_at_utc"] > d["kickoff_utc"]).any()):
        raise RuntimeError("MOM WFO contains a feature timestamp after kickoff")

    d["is_motm"] = pd.to_numeric(d["is_motm"], errors="coerce")
    if d["is_motm"].isna().any() or ~d["is_motm"].isin([0, 1]).all():
        raise RuntimeError("MOM WFO is_motm must be 0/1")
    counts = d.groupby("match_id", sort=False)["is_motm"].sum()
    if bool((counts != 1).any()):
        raise RuntimeError("MOM WFO requires exactly one positive MOTM label per match")

    values = d[list(MOM_FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise RuntimeError("MOM WFO feature matrix contains NaN or inf")

    return d.sort_values(["kickoff_utc", "match_id", "player_id"], kind="mergesort").reset_index(drop=True)


def _split_match_blocks(match_ids: list[str], n_blocks: int) -> list[list[str]]:
    if n_blocks < 3:
        raise ValueError("MOM WFO requires at least 3 blocks")
    if len(match_ids) < n_blocks:
        raise RuntimeError("Insufficient chronological matches for requested MOM WFO blocks")
    chunks = np.array_split(np.asarray(match_ids, dtype=object), n_blocks)
    blocks = [[str(x) for x in chunk.tolist()] for chunk in chunks if len(chunk)]
    if len(blocks) < 3:
        raise RuntimeError("MOM WFO produced fewer than 3 non-empty blocks")
    return blocks


def _ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(confidence) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidence >= lo) & (confidence < hi if i < bins - 1 else confidence <= hi)
        if mask.any():
            total += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(total)


def _evaluate_predictions(test: pd.DataFrame, distribution: pd.DataFrame) -> dict[str, float]:
    rows = []
    for match_id, group in distribution.groupby("match_id", sort=False):
        truth = test.loc[test["match_id"] == match_id]
        if truth.empty:
            raise RuntimeError(f"MOM WFO prediction has no truth rows for {match_id}")
        actual = str(truth.loc[truth["is_motm"] == 1, "player_id"].iloc[0])
        g = group.sort_values(["probability", "player_id"], ascending=[False, True], kind="mergesort")
        ids = g["player_id"].astype(str).tolist()
        probs = g["probability"].to_numpy(dtype=float)
        if len(ids) < 4:
            raise RuntimeError(f"MOM WFO match {match_id} has fewer than 4 candidates")
        if not np.isfinite(probs).all() or abs(float(probs.sum()) - 1.0) > 1e-9:
            raise RuntimeError(f"MOM WFO probabilities invalid for {match_id}")

        rank = ids.index(actual) + 1 if actual in ids else len(ids) + 1
        top1 = float(rank == 1)
        top4 = float(rank <= 4)
        reciprocal = float(1.0 / rank) if rank <= len(ids) else 0.0
        ndcg = float(1.0 / np.log2(rank + 1)) if rank <= 4 else 0.0
        true_index = ids.index(actual) if actual in ids else None
        true_prob = float(probs[true_index]) if true_index is not None else 1e-9
        y = np.zeros(len(probs), dtype=float)
        if true_index is not None:
            y[true_index] = 1.0
        brier = float(np.sum((probs - y) ** 2))
        rows.append({
            "top1": top1,
            "top4": top4,
            "mrr": reciprocal,
            "ndcg": ndcg,
            "logloss": -float(np.log(np.clip(true_prob, 1e-9, 1.0))),
            "brier": brier,
            "confidence": float(probs[0]),
            "correct": top1,
            "candidate_count": len(ids),
        })

    m = pd.DataFrame(rows)
    return {
        "top1_hit_rate": float(m["top1"].mean()),
        "top4_hit_rate": float(m["top4"].mean()),
        "mrr": float(m["mrr"].mean()),
        "ndcg_at_4": float(m["ndcg"].mean()),
        "logloss": float(m["logloss"].mean()),
        "brier": float(m["brier"].mean()),
        "ece": _ece(m["confidence"].to_numpy(), m["correct"].to_numpy()),
        "mean_candidate_count": float(m["candidate_count"].mean()),
        "min_candidate_count": int(m["candidate_count"].min()),
    }


def run_mom_walk_forward(
    history: pd.DataFrame,
    *,
    n_blocks: int = 6,
    locked_blocks: int = 2,
    min_train_matches: int = 10,
    regularization_c: float = 0.30,
    temperature: float = 1.0,
    random_state: int = 42,
    method: str = "binary_logit",
) -> pd.DataFrame:
    """Run expanding-window chronological OOS evaluation by complete match blocks."""
    d = _validate_history(history)
    match_order = (
        d[["match_id", "kickoff_utc"]]
        .drop_duplicates("match_id")
        .sort_values(["kickoff_utc", "match_id"], kind="mergesort")
        ["match_id"].astype(str).tolist()
    )
    blocks = _split_match_blocks(match_order, n_blocks)

    metrics: list[MOMBlockMetrics] = []
    for block_idx in range(1, len(blocks)):
        train_ids = [m for block in blocks[:block_idx] for m in block]
        test_ids = blocks[block_idx]
        if len(train_ids) < int(min_train_matches):
            continue

        train = d[d["match_id"].isin(train_ids)].copy()
        test = d[d["match_id"].isin(test_ids)].copy()

        model = fit_mom_model(
            train,
            regularization_c=regularization_c,
            temperature=temperature,
            min_matches=min_train_matches,
            random_state=random_state,
            method=method,
        )

        distributions = []
        for match_id, group in test.groupby("match_id", sort=False):
            kickoff = group["kickoff_utc"].iloc[0]
            prediction_time = kickoff - pd.Timedelta(seconds=1)
            pred_input = group.drop(columns=["is_motm"]).copy()
            dist = predict_mom_distribution(
                model,
                pred_input,
                prediction_time=prediction_time,
            )
            distributions.append(dist)

        if not distributions:
            continue
        dist = pd.concat(distributions, ignore_index=True)
        evaluated = _evaluate_predictions(test, dist)
        metrics.append(MOMBlockMetrics(
            block=block_idx,
            train_matches=len(train_ids),
            test_matches=len(test_ids),
            test_rows=len(test),
            **evaluated,
        ))

    if not metrics:
        raise RuntimeError("MOM WFO produced no evaluable OOS blocks")
    return pd.DataFrame([asdict(x) for x in metrics])


def split_mom_development_locked(
    block_metrics: pd.DataFrame,
    *,
    locked_blocks: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split already-computed chronological OOS blocks without touching metrics."""
    if block_metrics.empty:
        raise ValueError("MOM block metrics are empty")
    if len(block_metrics) <= int(locked_blocks):
        raise RuntimeError("MOM OOS requires development blocks before locked holdout")
    ordered = block_metrics.sort_values("block", kind="mergesort").reset_index(drop=True)
    development = ordered.iloc[:-int(locked_blocks)].copy()
    locked = ordered.iloc[-int(locked_blocks):].copy()
    if not bool((development["block"].max() < locked["block"].min())):
        raise RuntimeError("MOM development/locked blocks are not strictly separated")
    return development, locked
