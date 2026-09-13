from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.evaluation.metrics import classification_metrics
from src.models.baselines import candidates

TARGET_ACCURACY = 0.80


def _fit_predict(model, fit, target):
    model.fit(fit, target)
    return model


def _blend_weights(scores: dict[str, dict]) -> dict[str, float]:
    """Build a conservative soft ensemble from validation log-loss.

    The floor prevents one noisy historical validation window from receiving
    nearly all production weight. This reduces model-selection variance without
    adding another training pass.
    """
    names = list(scores)
    losses = np.array([max(float(scores[n]["logloss"]), 1e-6) for n in names], dtype=float)
    temperature = 0.08
    logits = -(losses - losses.min()) / temperature
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    floor = min(0.10, 1.0 / len(names))
    weights = np.maximum(weights, floor)
    weights /= weights.sum()
    return {name: float(w) for name, w in zip(names, weights)}


def _weighted_metric(parts: list[tuple[dict, int]], key: str) -> float:
    total = sum(n for _, n in parts)
    if not total:
        return float("nan")
    return float(sum(float(metric[key]) * n for metric, n in parts) / total)


def _validation_score(model, val: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Score older and recent validation slices without refitting the model.

    Both slices are strictly before the OOS block. The recent slice receives
    twice the weight, reflecting regime relevance while retaining stability from
    an older slice. This is deliberately cheap: each candidate is still fitted
    only once per walk-forward block.
    """
    if val.empty:
        raise ValueError("validation slice is empty")
    split = max(1, len(val) // 2)
    slices = [val.iloc[:split], val.iloc[split:]]
    parts = []
    for idx, sl in enumerate(slices):
        if sl.empty:
            continue
        metrics = classification_metrics(sl.target.astype(int), model.predict_proba(sl[feature_cols]))
        parts.append((metrics, len(sl) * (1 if idx == 0 else 2)))
    return {
        "logloss": _weighted_metric(parts, "logloss"),
        "accuracy": _weighted_metric(parts, "accuracy"),
        "brier": _weighted_metric(parts, "brier"),
        "ece": _weighted_metric(parts, "ece"),
    }


def run_walk_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_train: int = 1000,
    validation_frac: float = 0.2,
    oos_block: int | None = None,
    random_state: int = 42,
):
    """Chronological PIT-safe walk-forward evaluation.

    Model/ensemble selection uses only historical validation data. Validation is
    evaluated in two chronological slices, with extra weight on the recent slice,
    reducing sensitivity to one unusually easy or hard period. The immediately
    following OOS block is never used for selection, and the final two OOS blocks
    remain locked for adoption.
    """
    if oos_block is None:
        oos_block = max(500, int(os.getenv("SOCCER_OOS_BLOCK", "2000")))
    validation_max = max(500, int(os.getenv("SOCCER_VALIDATION_MAX", "2000")))
    d = df.sort_values("kickoff_utc", kind="mergesort").reset_index(drop=True).copy()
    d = d[d["pit_verified"] == True].reset_index(drop=True)
    d = d.dropna(subset=["target"])
    if len(d) < min_train + oos_block:
        raise ValueError(f"Not enough PIT-verified rows: {len(d)}; need at least {min_train + oos_block}")

    results, selected = [], []
    start = min_train
    while start < len(d):
        oos_end = min(start + oos_block, len(d))
        train = d.iloc[:start]
        oos = d.iloc[start:oos_end]
        val_n = min(max(60, int(len(train) * validation_frac)), validation_max, max(60, len(train) - 300))
        fit = train.iloc[:-val_n]
        val = train.iloc[-val_n:]
        Xfit, yfit = fit[feature_cols], fit.target.astype(int)

        scores = {}
        for name, model in candidates(random_state).items():
            _fit_predict(model, Xfit, yfit)
            scores[name] = _validation_score(model, val, feature_cols)
        weights = _blend_weights(scores)
        best = min(scores, key=lambda k: scores[k]["logloss"])
        selected.append({
            "oos_start": str(oos.kickoff_utc.min()),
            "selected_model": best,
            "blend": "recent_weighted_two_slice_validation_softmax",
            "weights": weights,
            "validation_logloss": scores[best]["logloss"],
            "validation_accuracy": scores[best]["accuracy"],
            "validation_brier": scores[best]["brier"],
            "validation_ece": scores[best]["ece"],
        })

        fitted = {
            name: _fit_predict(model, train[feature_cols], train.target.astype(int))
            for name, model in candidates(random_state).items()
        }
        probs = np.zeros((len(oos), 3), dtype=float)
        for name, model in fitted.items():
            probs += weights[name] * model.predict_proba(oos[feature_cols])
        probs = np.clip(probs, 1e-9, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
        candidate_metrics = classification_metrics(oos.target.astype(int), probs)
        baseline_model = fitted["logistic"]
        baseline_metrics = classification_metrics(oos.target.astype(int), baseline_model.predict_proba(oos[feature_cols]))
        results.append({
            "oos_start": str(oos.kickoff_utc.min()), "oos_end": str(oos.kickoff_utc.max()),
            "model": "ensemble", "best_single_model": best, **candidate_metrics,
            "baseline_logistic_logloss": baseline_metrics["logloss"], "baseline_logistic_accuracy": baseline_metrics["accuracy"],
            "baseline_logistic_brier": baseline_metrics["brier"], "baseline_logistic_ece": baseline_metrics["ece"],
            "n": len(oos), "target_accuracy": TARGET_ACCURACY,
            "target_met": bool(candidate_metrics["accuracy"] >= TARGET_ACCURACY), "target_gap": float(candidate_metrics["accuracy"] - TARGET_ACCURACY),
        })
        start = oos_end

    return pd.DataFrame(results), pd.DataFrame(selected)
