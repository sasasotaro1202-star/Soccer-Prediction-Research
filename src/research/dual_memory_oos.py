"""Research-only dual-memory probability challenger for soccer.

Combines a full-history prediction stream with a recent-history prediction
stream. The recent weight is learned only from already completed OOS folds.
No frozen holdout is used for selection and no production registry is touched.

Inputs are precomputed fold probabilities so this module remains model-agnostic.
"""
from __future__ import annotations

import numpy as np


def _safe_probs(p: np.ndarray) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError("probabilities must be finite non-negative shape (n,3)")
    arr = np.clip(arr, 1e-7, 1.0)
    return arr / arr.sum(axis=1, keepdims=True)


def _metric(y, p):
    y = np.asarray(y, dtype=int)
    q = _safe_probs(p)
    ll = float(-np.mean(np.log(q[np.arange(len(y)), y])))
    one = np.eye(3)[y]
    br = float(np.mean(np.sum((q - one) ** 2, axis=1)))
    return {
        "accuracy": float(np.mean(np.argmax(q, axis=1) == y)),
        "logloss": ll,
        "brier": br,
        "n": int(len(y)),
    }


def _soft_weight(history, *, temperature: float = 0.025) -> float:
    if not history:
        return 0.50
    recent_ll = float(np.mean([x["recent_logloss"] for x in history[-6:]]))
    full_ll = float(np.mean([x["full_logloss"] for x in history[-6:]]))
    recent_br = float(np.mean([x["recent_brier"] for x in history[-6:]]))
    full_br = float(np.mean([x["full_brier"] for x in history[-6:]]))
    gap = 0.70 * (full_ll - recent_ll) + 0.30 * (full_br - recent_br)
    raw = 1.0 / (1.0 + np.exp(-gap / max(temperature, 1e-6)))
    return float(np.clip(0.35 * 0.50 + 0.65 * raw, 0.20, 0.80))


def mix_probabilities(full_p, recent_p, recent_weight):
    full_q = _safe_probs(full_p)
    recent_q = _safe_probs(recent_p)
    if len(full_q) != len(recent_q):
        raise ValueError("full/recent prediction length mismatch")
    w = float(np.clip(recent_weight, 0.20, 0.80))
    out = (1.0 - w) * full_q + w * recent_q
    return _safe_probs(out)


def evaluate_dual_memory_oos(
    y: np.ndarray,
    folds: list[dict],
    *,
    min_blocks: int = 6,
) -> dict:
    history = []
    blocks = []
    for fold in folds:
        full_p = _safe_probs(fold["full"])
        recent_p = _safe_probs(fold["recent"])
        yf = np.asarray(y[int(fold["end"]):int(fold["te"])], dtype=int)
        if len(yf) != len(full_p) or len(yf) != len(recent_p):
            raise ValueError("fold target/prediction length mismatch")
        weight = _soft_weight(history)
        dynamic_p = mix_probabilities(full_p, recent_p, weight)
        full_m = _metric(yf, full_p)
        recent_m = _metric(yf, recent_p)
        dynamic_m = _metric(yf, dynamic_p)
        history.append({
            "full_logloss": full_m["logloss"],
            "recent_logloss": recent_m["logloss"],
            "full_brier": full_m["brier"],
            "recent_brier": recent_m["brier"],
        })
        blocks.append({
            "end": int(fold["end"]),
            "te": int(fold["te"]),
            "n": int(len(yf)),
            "recent_weight_before_block": weight,
            "full": full_m,
            "recent": recent_m,
            "dynamic": dynamic_m,
            "delta_dynamic_vs_full": {
                "accuracy": dynamic_m["accuracy"] - full_m["accuracy"],
                "logloss": dynamic_m["logloss"] - full_m["logloss"],
                "brier": dynamic_m["brier"] - full_m["brier"],
            },
            "delta_dynamic_vs_recent": {
                "accuracy": dynamic_m["accuracy"] - recent_m["accuracy"],
                "logloss": dynamic_m["logloss"] - recent_m["logloss"],
                "brier": dynamic_m["brier"] - recent_m["brier"],
            },
        })
    if len(blocks) < min_blocks:
        return {
            "status": "DEFERRED",
            "reason": "too_few_oos_blocks",
            "blocks": len(blocks),
            "production_changed": False,
            "promotion_evidence_eligible": False,
        }
    d = np.asarray([b["delta_dynamic_vs_full"]["logloss"] for b in blocks], dtype=float)
    br = np.asarray([b["delta_dynamic_vs_full"]["brier"] for b in blocks], dtype=float)
    ac = np.asarray([b["delta_dynamic_vs_full"]["accuracy"] for b in blocks], dtype=float)
    return {
        "status": "EVALUATED",
        "blocks": len(blocks),
        "samples": int(sum(b["n"] for b in blocks)),
        "mean_accuracy_delta_vs_full": float(ac.mean()),
        "mean_logloss_delta_vs_full": float(d.mean()),
        "mean_brier_delta_vs_full": float(br.mean()),
        "improved_logloss_ratio": float(np.mean(d < 0)),
        "improved_brier_ratio": float(np.mean(br < 0)),
        "non_worse_accuracy_ratio": float(np.mean(ac >= -0.005)),
        "final_recent_weight": float(_soft_weight(history)),
        "blocks_detail": blocks,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "holdout_used_for_selection": False,
        "policy": "chronological_oos_full_history_plus_recent_history_soft_routing",
    }
