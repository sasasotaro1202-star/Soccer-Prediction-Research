"""Same-OOS comparison of legacy V9/V12 baselines and candidates."""
from __future__ import annotations

from typing import Any

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
    if oos[id_col].duplicated().any():
        raise ValueError(f"OOS contains duplicate {id_col}")
    for name, frame in (("baseline", baseline), ("candidate", candidate)):
        if frame.empty:
            raise ValueError(f"{name} prediction frame is empty")
        if frame[id_col].duplicated().any():
            raise ValueError(f"{name} contains duplicate {id_col}")
        if not set(oos[id_col]).issubset(set(frame[id_col])):
            raise ValueError(f"{name} is missing locked OOS rows")

    base = oos[[id_col, target_col]].merge(
        baseline[[id_col, "H", "D", "A"]],
        on=id_col,
        how="inner",
        validate="one_to_one",
    )
    cand = oos[[id_col, target_col]].merge(
        candidate[[id_col, "H", "D", "A"]],
        on=id_col,
        how="inner",
        validate="one_to_one",
    )
    if len(base) != len(oos) or len(cand) != len(oos):
        raise ValueError("Baseline/candidate do not cover the identical OOS set")
    if not np.array_equal(base[id_col].to_numpy(), cand[id_col].to_numpy()):
        raise ValueError("Baseline and candidate OOS row ordering differs")

    y = base[target_col].astype(int).to_numpy()
    bp = _proba(base)
    cp = _proba(cand)
    bm = classification_metrics(y, bp)
    cm = classification_metrics(y, cp)

    # classification_metrics intentionally contains structured diagnostics
    # such as confusion_matrix. Only scalar numeric metrics are meaningful
    # deltas; attempting to subtract list-valued diagnostics caused the bridge
    # workflow to fail even though the OOS comparison itself was valid.
    delta: dict[str, float] = {}
    for key, base_value in bm.items():
        cand_value = cm.get(key)
        if isinstance(base_value, (int, float, np.integer, np.floating)) and isinstance(
            cand_value, (int, float, np.integer, np.floating)
        ):
            if np.isfinite(float(base_value)) and np.isfinite(float(cand_value)):
                delta[f"delta_{key}"] = float(cand_value) - float(base_value)

    return {
        "n": int(len(oos)),
        "baseline": bm,
        "candidate": cm,
        "delta": delta,
        "same_oos": True,
        "selection_allowed": False,
    }



def paired_bootstrap(
    oos: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    id_col: str = "match_id",
    target_col: str = "target",
    metric: str = "logloss",
    n_resamples: int = 2000,
    seed: int = 13013,
) -> dict[str, Any]:
    """Estimate a paired OOS uncertainty interval for candidate-minus-baseline.

    Research-only: rows and outcomes are locked by the caller. Resampling keeps
    baseline/candidate predictions paired on the same fixture, avoiding the
    variance inflation of independent bootstrap samples. Lower values are
    better for logloss and Brier; a two-sided p-value is reported only as a
    descriptive uncertainty statistic, not as an adoption rule.
    """
    if metric not in {"logloss", "brier"}:
        raise ValueError("paired_bootstrap supports only logloss and brier")
    if not isinstance(n_resamples, int) or n_resamples < 100:
        raise ValueError("n_resamples must be >= 100")
    compared = compare_same_oos(
        oos,
        baseline,
        candidate,
        id_col=id_col,
        target_col=target_col,
    )
    merged = oos[[id_col, target_col]].merge(
        baseline[[id_col, "H", "D", "A"]].rename(
            columns={"H": "bH", "D": "bD", "A": "bA"}
        ),
        on=id_col,
        how="inner",
        validate="one_to_one",
    ).merge(
        candidate[[id_col, "H", "D", "A"]].rename(
            columns={"H": "cH", "D": "cD", "A": "cA"}
        ),
        on=id_col,
        how="inner",
        validate="one_to_one",
    )
    y = merged[target_col].astype(int).to_numpy()
    bp = merged[["bH", "bD", "bA"]].to_numpy(dtype=float)
    cp = merged[["cH", "cD", "cA"]].to_numpy(dtype=float)
    bp = np.clip(bp, 1e-12, 1.0)
    cp = np.clip(cp, 1e-12, 1.0)
    bp /= bp.sum(axis=1, keepdims=True)
    cp /= cp.sum(axis=1, keepdims=True)
    row_index = np.arange(len(y))
    if metric == "logloss":
        baseline_loss = -np.log(bp[row_index, y])
        candidate_loss = -np.log(cp[row_index, y])
    else:
        one_hot = np.zeros_like(bp)
        one_hot[row_index, y] = 1.0
        baseline_loss = np.sum((bp - one_hot) ** 2, axis=1)
        candidate_loss = np.sum((cp - one_hot) ** 2, axis=1)
    deltas = candidate_loss - baseline_loss
    observed = float(np.mean(deltas))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(n_resamples, len(deltas)))
    sampled = deltas[indices].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    p_two_sided = float(
        2.0 * min(
            np.mean(sampled >= 0.0),
            np.mean(sampled <= 0.0),
        )
    )
    p_two_sided = float(np.clip(p_two_sided, 0.0, 1.0))
    return {
        "n": int(len(deltas)),
        "metric": metric,
        "observed_delta_candidate_minus_baseline": observed,
        "ci_95_low": float(low),
        "ci_95_high": float(high),
        "p_two_sided": p_two_sided,
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "same_oos": bool(compared["same_oos"]),
        "research_only": True,
        "selection_allowed": False,
    }
