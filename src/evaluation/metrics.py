from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss


def multiclass_brier(y_true, proba):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    onehot = np.zeros_like(p, dtype=float)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def ranked_probability_score(y_true, proba):
    """Multiclass RPS for ordered H/D/A outcomes (lower is better)."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    p = np.clip(p, 1e-9, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    cumulative = np.cumsum(p, axis=1)[:, :-1]
    truth = np.zeros_like(p, dtype=float)
    truth[np.arange(len(y)), y] = 1.0
    cumulative_truth = np.cumsum(truth, axis=1)[:, :-1]
    k = max(1, p.shape[1] - 1)
    return float(np.mean(np.sum((cumulative - cumulative_truth) ** 2, axis=1) / k))


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
    """Return proper scores plus class support/precision/recall diagnostics.

    Class-specific hit rate is explicitly defined as recall among matches whose
    true class is that outcome. This prevents the metric from being mistaken for
    overall accuracy or prediction share. The confusion matrix uses fixed H/D/A
    labels so missing classes never change the column/row meaning.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("proba must have shape (n, 3) for H/D/A")
    p = np.clip(p, 1e-9, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    pred = p.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    out = {
        "logloss": float(log_loss(y, p, labels=[0, 1, 2])),
        "accuracy": float(accuracy_score(y, pred)),
        "brier": multiclass_brier(y, p),
        "rps": ranked_probability_score(y, p),
        "ece": ece(y, p),
        "n": int(len(y)),
        "actual_home_n": int((y == 0).sum()),
        "actual_draw_n": int((y == 1).sum()),
        "actual_away_n": int((y == 2).sum()),
        "pred_home_n": int((pred == 0).sum()),
        "pred_draw_n": int((pred == 1).sum()),
        "pred_away_n": int((pred == 2).sum()),
    }
    for cls, name in enumerate(("home", "draw", "away")):
        mask = y == cls
        support = int(mask.sum())
        predicted = int((pred == cls).sum())
        tp = int(cm[cls, cls])
        out[f"{name}_support"] = support
        out[f"{name}_logloss"] = float(-np.log(p[mask, cls]).mean()) if support else float("nan")
        out[f"{name}_prob_mean"] = float(p[:, cls].mean())
        out[f"{name}_hit_rate"] = float(tp / support) if support else float("nan")
        out[f"{name}_precision"] = float(tp / predicted) if predicted else float("nan")
    out["macro_recall"] = float(np.mean([out[f"{n}_hit_rate"] for n in ("home", "draw", "away") if np.isfinite(out[f"{n}_hit_rate"])]))
    out["confusion_matrix"] = cm.tolist()
    return out
