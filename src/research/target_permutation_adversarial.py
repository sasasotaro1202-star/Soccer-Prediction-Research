"""Research-only adversarial target-permutation audit for soccer OOS pipelines.

The audit retrains the same model family after shuffling only training targets.
Evaluation features and labels remain untouched. It is a diagnostic for detecting
suspicious target-bearing features or leakage; it is never a production gate.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

Array = np.ndarray
FitPredict = Callable[[Array, Array, Array, int], Array]


def _metrics(y: Array, p: Array) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or len(p) != len(y) or p.shape[1] != 3:
        raise ValueError("soccer target-permutation probabilities must have shape (n, 3)")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    sums = p.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("probability rows must have positive sums")
    p = p / sums
    idx = np.arange(len(y))
    logloss = -float(np.mean(np.log(np.clip(p[idx, y], 1e-15, 1.0))))
    one_hot = np.zeros_like(p)
    one_hot[idx, y] = 1.0
    brier = float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))
    accuracy = float(np.mean(np.argmax(p, axis=1) == y))
    return {"logloss": logloss, "brier": brier, "accuracy": accuracy}


def audit_target_permutation(
    *,
    X_train: Array,
    y_train: Array,
    X_eval: Array,
    y_eval: Array,
    fit_predict: FitPredict,
    seeds: Sequence[int] = (7, 19, 43, 71, 101, 137),
) -> dict[str, object]:
    """Compare a real-target model against shuffled-target null retraining.

    A strong real-vs-null separation is expected for an ordinary predictive
    dataset. If the shuffled-target models remain implausibly competitive,
    investigate target-bearing features, split leakage, caching, preprocessing,
    or label contamination.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train, dtype=int)
    X_eval = np.asarray(X_eval)
    y_eval = np.asarray(y_eval, dtype=int)
    if len(X_train) != len(y_train) or len(X_eval) != len(y_eval):
        raise ValueError("target-permutation train/eval row mismatch")
    if len(X_train) < 30 or len(X_eval) < 10:
        raise ValueError("target-permutation audit requires at least 30 train and 10 eval rows")
    if np.unique(y_train).size < 2 or np.any((y_train < 0) | (y_train > 2)):
        raise ValueError("soccer training targets must contain H/D/A classes encoded 0/1/2")
    if np.any((y_eval < 0) | (y_eval > 2)):
        raise ValueError("soccer evaluation targets must be encoded 0/1/2")
    seed_list = tuple(int(s) for s in seeds)
    if len(seed_list) < 3:
        raise ValueError("at least 3 permutation seeds are required")

    real = _metrics(y_eval, fit_predict(X_train, y_train, X_eval, seed_list[0]))
    nulls: list[dict[str, float]] = []
    for seed in seed_list:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(y_train)
        nulls.append(_metrics(y_eval, fit_predict(X_train, shuffled, X_eval, seed)))

    null_ll = np.asarray([x["logloss"] for x in nulls], dtype=float)
    null_brier = np.asarray([x["brier"] for x in nulls], dtype=float)
    null_acc = np.asarray([x["accuracy"] for x in nulls], dtype=float)

    separation = {
        "logloss_null_mean_minus_real": float(null_ll.mean() - real["logloss"]),
        "brier_null_mean_minus_real": float(null_brier.mean() - real["brier"]),
        "accuracy_real_minus_null_mean": float(real["accuracy"] - null_acc.mean()),
    }
    strong = (
        real["logloss"] < float(np.quantile(null_ll, 0.10))
        and real["brier"] < float(np.quantile(null_brier, 0.10))
        and real["accuracy"] > float(np.quantile(null_acc, 0.90))
    )
    suspicious = (
        real["logloss"] >= float(null_ll.mean() - 0.01)
        or real["brier"] >= float(null_brier.mean() - 0.01)
        or real["accuracy"] <= float(null_acc.mean() + 0.02)
    )
    status = "SEPARATED" if strong and not suspicious else ("SUSPICIOUS" if suspicious else "NO_SEPARATION")

    return {
        "status": status,
        "risk_flag": bool(suspicious),
        "seeds": list(seed_list),
        "real": real,
        "null_summary": {
            "logloss_mean": float(null_ll.mean()),
            "logloss_q10": float(np.quantile(null_ll, 0.10)),
            "logloss_q90": float(np.quantile(null_ll, 0.90)),
            "brier_mean": float(null_brier.mean()),
            "brier_q10": float(np.quantile(null_brier, 0.10)),
            "brier_q90": float(np.quantile(null_brier, 0.90)),
            "accuracy_mean": float(null_acc.mean()),
            "accuracy_q10": float(np.quantile(null_acc, 0.10)),
            "accuracy_q90": float(np.quantile(null_acc, 0.90)),
            "permutations": int(len(nulls)),
        },
        "separation": separation,
        "research_only": True,
        "production_changed": False,
        "selection_allowed": False,
        "interpretation": "A suspicious result triggers investigation; it does not prove leakage.",
    }
