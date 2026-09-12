from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import classification_metrics
from src.models.baselines import candidates

TARGET_ACCURACY = 0.80


def _fit_predict(model, fit, target):
    model.fit(fit, target)
    return model


def _blend_weights(scores: dict[str, dict]) -> dict[str, float]:
    # LogLoss remains the primary objective; inverse-loss weights are learned only
    # from the historical validation slice and are never fitted on the OOS block.
    losses = np.array([max(float(v["logloss"]), 1e-6) for v in scores.values()])
    inv = 1.0 / losses
    inv /= inv.sum()
    return {name: float(w) for name, w in zip(scores, inv)}


def run_walk_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_train: int = 1000,
    validation_frac: float = 0.2,
    oos_block: int = 500,
    random_state: int = 42,
):
    d = df.sort_values("kickoff_utc").reset_index(drop=True).copy()
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
        val_n = min(max(60, int(len(train) * validation_frac)), max(60, len(train) - 300))
        fit = train.iloc[:-val_n]
        val = train.iloc[-val_n:]
        Xfit, yfit = fit[feature_cols], fit.target.astype(int)
        Xval, yval = val[feature_cols], val.target.astype(int)

        scores = {}
        for name, model in candidates(random_state).items():
            _fit_predict(model, Xfit, yfit)
            scores[name] = classification_metrics(yval, model.predict_proba(Xval))
        weights = _blend_weights(scores)
        best = min(scores, key=lambda k: scores[k]["logloss"])
        selected.append({
            "oos_start": str(oos.kickoff_utc.min()),
            "selected_model": best,
            "blend": "validation_inverse_logloss",
            "weights": weights,
            "validation_logloss": scores[best]["logloss"],
            "validation_accuracy": scores[best]["accuracy"],
        })

        fitted = {}
        for name, model in candidates(random_state).items():
            fitted[name] = _fit_predict(model, train[feature_cols], train.target.astype(int))
        probs = np.zeros((len(oos), 3), dtype=float)
        for name, model in fitted.items():
            probs += weights[name] * model.predict_proba(oos[feature_cols])
        probs = np.clip(probs, 1e-9, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
        m = classification_metrics(oos.target.astype(int), probs)
        result = {
            "oos_start": str(oos.kickoff_utc.min()), "oos_end": str(oos.kickoff_utc.max()),
            "model": "ensemble", "best_single_model": best, **m, "n": len(oos),
            "target_accuracy": TARGET_ACCURACY, "target_met": bool(m["accuracy"] >= TARGET_ACCURACY),
            "target_gap": float(m["accuracy"] - TARGET_ACCURACY),
        }
        results.append(result)
        start = oos_end

    return pd.DataFrame(results), pd.DataFrame(selected)
