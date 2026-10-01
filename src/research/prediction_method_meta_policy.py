"""Research-only prediction-method meta-policy for soccer.

The policy chooses a *decision representation* rather than changing the model
probabilities. It is outcome-free and intended to be evaluated chronologically
before any production adoption.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


POLICY_VERSION = "soccer-prediction-method-meta-policy-v1"
HIGH_CONFIDENCE = 0.80
MEDIUM_CONFIDENCE = 0.65
HIGH_DISAGREEMENT = 0.05
HIGH_RISK = 0.66
LOW_PREDICTABILITY = 0.45
MEDIUM_PREDICTABILITY = 0.65


def _policy_hash() -> str:
    payload = {
        "policy_version": POLICY_VERSION,
        "thresholds": {
            "high_confidence": HIGH_CONFIDENCE,
            "medium_confidence": MEDIUM_CONFIDENCE,
            "high_disagreement": HIGH_DISAGREEMENT,
            "high_risk": HIGH_RISK,
            "low_predictability": LOW_PREDICTABILITY,
            "medium_predictability": MEDIUM_PREDICTABILITY,
        },
        "principles": {
            "probability_unchanged": True,
            "outcome_free": True,
            "pit_required": True,
            "automatic_promotion": False,
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _probability_state(probabilities: Sequence[float]) -> tuple[np.ndarray, float, float]:
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Soccer 1X2 probabilities must be finite and shape (3,)")
    total = float(p.sum())
    if total <= 0:
        raise ValueError("Soccer probabilities must have positive total mass")
    p = p / total
    ordered = np.sort(p)
    confidence = float(ordered[-1])
    margin = float(ordered[-1] - ordered[-2])
    return p, confidence, margin


def _predictability(
    raw: float | None,
    *,
    disagreement: float,
    entropy: float | None,
    risk: float | None,
) -> tuple[float | None, bool]:
    if raw is not None:
        value = float(raw)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("raw_predictability must be in [0,1]")
        return value, False

    components = []
    weights = []
    if np.isfinite(disagreement):
        components.append(np.clip(float(disagreement) / 0.10, 0.0, 1.0))
        weights.append(0.35)
    if entropy is not None:
        ent = float(entropy)
        if not np.isfinite(ent) or ent < 0:
            raise ValueError("entropy must be finite and non-negative")
        # 3-way normalized entropy is approximately [0,1].
        components.append(np.clip(ent, 0.0, 1.0))
        weights.append(0.40)
    if risk is not None:
        rv = float(risk)
        if not np.isfinite(rv) or not 0.0 <= rv <= 1.0:
            raise ValueError("risk must be in [0,1]")
        components.append(rv)
        weights.append(0.25)
    if not components:
        return None, True

    score = 1.0 - float(np.dot(np.asarray(components), np.asarray(weights)) / max(sum(weights), 1e-12))
    return float(np.clip(score, 0.0, 1.0)), True


def select_method(
    *,
    probabilities: Sequence[float],
    strategy: str,
    router_status: str,
    pit_safe: bool,
    raw_predictability: float | None = None,
    model_disagreement: float = 0.0,
    predictive_entropy: float | None = None,
    risk_score: float | None = None,
    score_distribution_available: bool = False,
    mom_available: bool = False,
    freshness_score: float | None = None,
) -> dict[str, Any]:
    """Return a typed decision policy without modifying prediction probabilities."""
    p, confidence, margin = _probability_state(probabilities)
    if not pit_safe:
        return {
            "schema_version": 1,
            "policy_version": POLICY_VERSION,
            "policy_hash": _policy_hash(),
            "status": "FAIL_CLOSED",
            "method": {"model_method": strategy, "router_status": router_status},
            "probability_changed": False,
            "target": {"type": "multiclass_1x2", "granularity": "event", "horizon": "pre_kickoff"},
            "uncertainty": {"level": "UNKNOWN", "predictability": None},
            "output": {"format_candidate": "abstain_or_fallback", "action": "ABSTAIN"},
            "safety": {"pit_required": True, "production_auto_promotion": False},
        }

    predictability, derived = _predictability(
        raw_predictability,
        disagreement=model_disagreement,
        entropy=predictive_entropy,
        risk=risk_score,
    )
    high_risk = risk_score is not None and float(risk_score) >= HIGH_RISK
    high_disagreement = float(model_disagreement) >= HIGH_DISAGREEMENT
    stale = freshness_score is not None and float(freshness_score) < 0.50
    high_conf_low_pred = confidence >= HIGH_CONFIDENCE and predictability is not None and predictability < LOW_PREDICTABILITY

    if high_conf_low_pred or high_risk or high_disagreement:
        uncertainty_level = "HIGH"
        output_format = "prediction_set_or_scenario" if score_distribution_available else "abstain_or_fallback"
        action = "REVIEW"
    elif predictability is not None and predictability < MEDIUM_PREDICTABILITY:
        uncertainty_level = "MEDIUM"
        output_format = "probability_distribution"
        action = "SECONDARY"
    else:
        uncertainty_level = "LOW"
        output_format = "probability_distribution"
        action = "PRIMARY" if confidence >= MEDIUM_CONFIDENCE else "SECONDARY"

    next_action = (
        "refresh_pit_safe_sources"
        if stale
        else "recompute_candidate"
        if high_disagreement or high_risk
        else "maintain_until_cutoff"
    )

    return {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "policy_hash": _policy_hash(),
        "status": "SELECTED",
        "target": {
            "type": "multiclass_1x2",
            "granularity": "event",
            "horizon": "pre_kickoff",
        },
        "method": {
            "model_method": str(strategy),
            "router_status": str(router_status),
            "model_route_changed": False,
        },
        "probability": {
            "home": float(p[0]),
            "draw": float(p[1]),
            "away": float(p[2]),
            "confidence": confidence,
            "margin": margin,
            "probability_changed": False,
        },
        "uncertainty": {
            "level": uncertainty_level,
            "predictability": predictability,
            "predictability_derived": bool(derived),
            "model_disagreement": float(np.clip(model_disagreement, 0.0, 1.0)),
            "risk_score": None if risk_score is None else float(np.clip(risk_score, 0.0, 1.0)),
            "high_confidence_low_predictability": bool(high_conf_low_pred),
        },
        "output": {
            "format_candidate": output_format,
            "action": action,
            "score_distribution_available": bool(score_distribution_available),
            "mom_available": bool(mom_available),
        },
        "information": {
            "next_action": next_action,
            "pit_safe_only": True,
            "freshness_score": freshness_score,
        },
        "safety": {
            "pit_required": True,
            "outcome_free": True,
            "production_auto_promotion": False,
            "probability_generation_untouched": True,
        },
    }


def validate_decision(decision: Mapping[str, Any]) -> None:
    if decision.get("policy_version") != POLICY_VERSION:
        raise ValueError("unknown policy version")
    if decision.get("status") not in {"SELECTED", "FAIL_CLOSED"}:
        raise ValueError("invalid policy status")
    safety = decision.get("safety") or {}
    if safety.get("production_auto_promotion") is not False:
        raise ValueError("meta-policy must not auto-promote to production")
    if safety.get("probability_generation_untouched") is not True and decision.get("status") == "SELECTED":
        raise ValueError("meta-policy must not modify probability generation")
