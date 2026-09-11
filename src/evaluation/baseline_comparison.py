"""Same-OOS comparison of legacy V9/V12 baselines and candidates."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.evaluation.metrics import classification_metrics


def _proba(predictions: pd.DataFrame) -> np.ndarray:
    required = ["H", "D", "A"]
    if not all(c in predictions.columns for c in required):
        raise ValueError("Prediction frame must contain H, D, A probability columns")
    p = predictions[required].to_numpy(dtype=float)
    if len(p) == 0:
        raise ValueError("Prediction frame is empty")
    if not np.all(np.isfinite(p)):
        raise ValueError("Prediction probabilities contain non-finite values")
    if np.any(p <= 0):
        raise ValueError("Prediction probabilities must be strictly positive")
    sums = p.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-6):
        raise ValueError("Prediction probabilities must sum to 1")
    return p


def compare_same_oos(
    oos: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    id_col: str = "match_id",
    target_col: str = "target",
) -> dict[str, Any]:
    """Compare two models on exactly the same locked OOS rows.

    The function rejects duplicate IDs, target disagreement, missing rows, or
    different OOS sets. It never selects the OOS period itself.
    """
    if oos.empty:
        raise ValueError("OOS frame is empty")
    for name, frame in (("baseline", baseline), ("candidate", candidate)):
        if frame.empty:
            raise ValueError(f"{name} prediction frame is empty")
        if frame[id_col].duplicated().any():
            raise ValueError(f"{name} contains duplicate {id_col}")
        if not set(oos[id_col]).issubset(set(frame[id_col])):
            raise ValueError(f"{name} is missing locked OOS rows")

    base = oos[[id_col, target_col]].merge(baseline[[id_col, "H", "D", "A"]], on=id_col, how="inner", validate="one_to_one")
    cand = oos[[id_col, target_col]].merge(candidate[[id_col, "H", "D", "A"]], on=id_col, how="inner", validate="one_to_one")
    if len(base) != len(oos) or len(cand) != len(oos):
        raise ValueError("Baseline/candidate do not cover the identical OOS set")
    if not np.array_equal(base[id_col].to_numpy(), cand[id_col].to_numpy()):
        raise ValueError("Baseline and candidate OOS row ordering differs")
    y = base[target_col].astype(int).to_numpy()
    bp = _proba(base)
    cp = _proba(cand)
    bm = classification_metrics(y, bp)
    cm = classification_metrics(y, cp)
    delta = {f"delta_{k}": float(cm[k] - bm[k]) for k in bm}
    return {
        "n": int(len(oos)),
        "baseline": bm,
        "candidate": cm,
        "delta": delta,
        "same_oos": True,
        "selection_allowed": False,
    }
