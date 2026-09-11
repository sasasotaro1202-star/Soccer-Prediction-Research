"""Independent-OOS adoption gate for legacy-vs-candidate promotion."""
from __future__ import annotations

from typing import Any, Mapping


def _better(candidate: Mapping[str, float], baseline: Mapping[str, float], key: str, lower: bool) -> bool:
    return float(candidate[key]) < float(baseline[key]) if lower else float(candidate[key]) > float(baseline[key])


def independent_adoption_gate(
    development: Mapping[str, Any],
    holdout: Mapping[str, Any],
    *,
    min_holdout_rows: int = 100,
) -> dict[str, Any]:
    """Require improvement over the legacy baseline on locked independent OOS.

    Development OOS may be used to generate/select a candidate. The holdout is
    never used for selection. Promotion requires primary LogLoss improvement,
    no regression in Brier/ECE/Accuracy, sufficient holdout rows, and explicit
    same-OOS evidence.
    """
    if int(holdout.get("n", 0)) < min_holdout_rows:
        return {"status": "HOLD", "reason": "independent_holdout_too_small"}
    if holdout.get("same_oos") is not True:
        return {"status": "HOLD", "reason": "holdout_is_not_same_oos"}

    base = holdout.get("baseline", {})
    cand = holdout.get("candidate", {})
    required = ("logloss", "brier", "ece", "accuracy")
    if not all(k in base and k in cand for k in required):
        return {"status": "HOLD", "reason": "incomplete_holdout_metrics"}

    primary = _better(cand, base, "logloss", True)
    secondary = (
        _better(cand, base, "brier", True)
        and _better(cand, base, "ece", True)
        and _better(cand, base, "accuracy", False)
    )
    return {
        "status": "ADOPT" if primary and secondary else "REJECT",
        "primary_logloss_improved": primary,
        "secondary_ok": secondary,
        "development_evidence_present": bool(development),
        "holdout_rows": int(holdout["n"]),
        "promotion_authority": "deterministic_research_engine",
    }
