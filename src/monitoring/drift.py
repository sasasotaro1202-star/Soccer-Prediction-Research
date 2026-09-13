from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _safe_hist(values: np.ndarray, edges: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    p = counts.astype(float) + eps
    return p / p.sum()


def psi(reference: Iterable[float], current: Iterable[float], bins: int = 10) -> float:
    """Population Stability Index for numeric distributions.

    Bins are learned from the reference sample only. This function is monitoring,
    not model fitting; no current-sample information is used to define cut points.
    """
    ref = pd.to_numeric(pd.Series(reference), errors="coerce").dropna().to_numpy(float)
    cur = pd.to_numeric(pd.Series(current), errors="coerce").dropna().to_numpy(float)
    if len(ref) < 20 or len(cur) < 20:
        return float("nan")
    quantiles = np.linspace(0.0, 1.0, max(2, int(bins)) + 1)
    edges = np.quantile(ref, quantiles)
    edges = np.unique(edges)
    if len(edges) < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    p = _safe_hist(ref, edges)
    q = _safe_hist(cur, edges)
    return float(np.sum((q - p) * np.log(q / p)))


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Return distribution and missingness diagnostics for shared numeric features."""
    rows = []
    for col in sorted(set(reference.columns) & set(current.columns)):
        r = pd.to_numeric(reference[col], errors="coerce")
        c = pd.to_numeric(current[col], errors="coerce")
        if r.notna().sum() < 20 and c.notna().sum() < 20:
            continue
        rows.append({
            "feature": col,
            "reference_n": int(r.notna().sum()),
            "current_n": int(c.notna().sum()),
            "reference_missing_rate": float(r.isna().mean()),
            "current_missing_rate": float(c.isna().mean()),
            "reference_mean": float(r.mean()) if r.notna().any() else np.nan,
            "current_mean": float(c.mean()) if c.notna().any() else np.nan,
            "reference_std": float(r.std(ddof=0)) if r.notna().any() else np.nan,
            "current_std": float(c.std(ddof=0)) if c.notna().any() else np.nan,
            "psi": psi(r, c),
        })
    return pd.DataFrame(rows)


def prediction_drift(reference_prob: pd.DataFrame, current_prob: pd.DataFrame) -> dict:
    """Monitor class probability drift without using labels."""
    cols = [c for c in reference_prob.columns if c in current_prob.columns]
    if not cols:
        return {"status": "NO_SHARED_PROBABILITY_COLUMNS"}
    out = {"status": "OK", "columns": {}, "reference_n": int(len(reference_prob)), "current_n": int(len(current_prob))}
    for col in cols:
        r = pd.to_numeric(reference_prob[col], errors="coerce").dropna()
        c = pd.to_numeric(current_prob[col], errors="coerce").dropna()
        if len(r) < 20 or len(c) < 20:
            out["columns"][col] = {"status": "INSUFFICIENT_SAMPLE"}
            continue
        out["columns"][col] = {
            "reference_mean": float(r.mean()),
            "current_mean": float(c.mean()),
            "reference_std": float(r.std(ddof=0)),
            "current_std": float(c.std(ddof=0)),
            "psi": psi(r, c),
        }
    return out
