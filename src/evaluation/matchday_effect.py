from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import classification_metrics

PROBABILITY_COLUMNS = {
    "baseline": ("base_p_home", "base_p_draw", "base_p_away"),
    "matchday": ("p_home", "p_draw", "p_away"),
}

def _validated_probability_frame(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    columns = PROBABILITY_COLUMNS[prefix]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction frame missing {prefix} probability columns: {missing}")
    p = frame[list(columns)].to_numpy(dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError(f"{prefix} probabilities are invalid")
    if np.any(p <= 0) or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(f"{prefix} probabilities must be strictly positive and sum to one")
    return p

def evaluate_matchday_effect(predictions: pd.DataFrame, results: pd.DataFrame, *, id_col: str = "match_id", target_col: str = "target") -> dict[str, Any]:
    """Measure matchday adjustment effect on the exact same settled fixtures."""
    required_pred = {id_col, *PROBABILITY_COLUMNS["baseline"], *PROBABILITY_COLUMNS["matchday"]}
    required_res = {id_col, target_col}
    if missing := sorted(required_pred - set(predictions.columns)):
        raise ValueError(f"Prediction frame missing columns: {missing}")
    if missing := sorted(required_res - set(results.columns)):
        raise ValueError(f"Results frame missing columns: {missing}")
    if predictions[id_col].duplicated().any():
        raise ValueError("Predictions contain duplicate match IDs")
    if results[id_col].duplicated().any():
        raise ValueError("Results contain duplicate match IDs")

    paired = predictions.merge(results[[id_col, target_col]], on=id_col, how="inner", validate="one_to_one")
    if paired.empty:
        raise ValueError("No settled prediction rows overlap results")
    y = pd.to_numeric(paired[target_col], errors="coerce")
    if y.isna().any() or not y.isin([0, 1, 2]).all():
        raise ValueError("Results target must contain only H/D/A class IDs 0,1,2")
    yv = y.astype(int).to_numpy()
    base = _validated_probability_frame(paired, "baseline")
    final = _validated_probability_frame(paired, "matchday")

    bm = classification_metrics(yv, base)
    fm = classification_metrics(yv, final)

    row_base = -np.log(np.clip(base[np.arange(len(yv)), yv], 1e-15, 1.0))
    row_final = -np.log(np.clip(final[np.arange(len(yv)), yv], 1e-15, 1.0))
    truth = np.eye(3, dtype=float)[yv]
    brier_base = np.sum((base - truth) ** 2, axis=1)
    brier_final = np.sum((final - truth) ** 2, axis=1)
    changed = np.max(np.abs(base - final), axis=1) > 1e-12

    per_match = pd.DataFrame({
        id_col: paired[id_col].astype(str).to_numpy(),
        "matchday_applied": paired.get("matchday_applied", pd.Series(False, index=paired.index)).astype(bool).to_numpy(),
        "matchday_status": paired.get("matchday_status", pd.Series("UNKNOWN", index=paired.index)).astype(str).to_numpy(),
        "logloss_delta": row_final - row_base,
        "brier_delta": brier_final - brier_base,
        "baseline_prediction": np.argmax(base, axis=1),
        "matchday_prediction": np.argmax(final, axis=1),
    })

    out: dict[str, Any] = {
        "n": int(len(paired)),
        "changed_n": int(changed.sum()),
        "overall": {
            "baseline": bm,
            "matchday": fm,
            "delta": {
                "logloss": float(fm["logloss"] - bm["logloss"]),
                "brier": float(fm["brier"] - bm["brier"]),
                "rps": float(fm["rps"] - bm["rps"]),
                "ece": float(fm["ece"] - bm["ece"]),
                "accuracy": float(fm["accuracy"] - bm["accuracy"]),
            },
        },
        "per_match": per_match,
    }

    for key, mask in {"applied": changed, "unchanged": ~changed}.items():
        if not np.any(mask):
            out[key] = {"n": 0}
        else:
            out[key] = {
                "n": int(mask.sum()),
                "baseline": classification_metrics(yv[mask], base[mask]),
                "matchday": classification_metrics(yv[mask], final[mask]),
            }
    return out

def summarize_matchday_effect(result: dict[str, Any]) -> pd.DataFrame:
    frame = result.get("per_match")
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Result does not contain a per-match attribution table")
    return frame.copy()
