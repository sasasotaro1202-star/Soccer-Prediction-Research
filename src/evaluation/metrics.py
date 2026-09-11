from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, log_loss


def multiclass_brier(y_true, proba):
    y = np.asarray(y_true)
    p = np.asarray(proba)
    onehot = np.zeros_like(p, dtype=float)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def ece(y_true, proba, bins=10):
    y = np.asarray(y); p = np.asarray(proba)
    conf = p.max(axis=1); pred = p.argmax(axis=1); correct = (pred == y)
    total = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(0, 1, bins + 1)[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if m.any(): total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def classification_metrics(y_true, proba):
    p = np.asarray(proba)
    pred = p.argmax(axis=1)
    return {
        "logloss": float(log_loss(y_true, p, labels=[0, 1, 2])),
        "accuracy": float(accuracy_score(y_true, pred)),
        "brier": multiclass_brier(y_true, p),
        "ece": ece(y_true, p),
    }
