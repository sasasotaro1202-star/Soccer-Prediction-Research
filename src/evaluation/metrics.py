from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, log_loss


def multiclass_brier(y_true, proba):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    onehot = np.zeros_like(p, dtype=float)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def ece(y_true, proba, bins=10):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = pred == y
    total = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & ((conf < hi) if i < bins - 1 else (conf <= hi))
        if mask.any():
            total += mask.mean() * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(total)


def classification_metrics(y_true, proba):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    p = np.clip(p, 1e-9, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    pred = p.argmax(axis=1)
    out = {
        "logloss": float(log_loss(y, p, labels=[0, 1, 2])),
        "accuracy": float(accuracy_score(y, pred)),
        "brier": multiclass_brier(y, p),
        "ece": ece(y, p),
    }
    for cls, name in enumerate(("home", "draw", "away")):
        mask = y == cls
        out[f"{name}_logloss"] = float(-np.log(p[mask, cls]).mean()) if mask.any() else float("nan")
        out[f"{name}_prob_mean"] = float(p[:, cls].mean())
        out[f"{name}_hit_rate"] = float((pred[mask] == cls).mean()) if mask.any() else float("nan")
    return out
