"""Independent-OOS adoption gate for legacy-vs-candidate promotion."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping


def _better(candidate: Mapping[str, float], baseline: Mapping[str, float], key: str, lower: bool) -> bool:
    return float(candidate[key]) < float(baseline[key]) if lower else float(candidate[key]) > float(baseline[key])


def _parse_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone()


def _holdout_integrity(holdout: Mapping[str, Any]) -> tuple[bool, str]:
    """Fail closed unless the holdout is explicitly locked and selection-independent.

    The additional metadata is deliberately mandatory for promotion but optional for
    legacy callers at the type level. Missing metadata therefore blocks adoption
    rather than silently treating an unverified split as an independent holdout.
    """
    if holdout.get("locked") is not True:
        return False, "holdout_not_explicitly_locked"
    if holdout.get("selection_frozen") is not True:
        return False, "holdout_selection_not_frozen"
    if holdout.get("used_for_selection") is True:
        return False, "holdout_was_used_for_selection"
    if holdout.get("used_for_calibration") is True:
        return False, "holdout_was_used_for_calibration"
    if holdout.get("used_for_threshold_tuning") is True:
        return False, "holdout_was_used_for_threshold_tuning"

    development_end = _parse_utc(holdout.get("development_end_utc"))
    holdout_start = _parse_utc(holdout.get("holdout_start_utc"))
    if development_end is None or holdout_start is None:
        return False, "holdout_temporal_boundaries_missing_or_invalid"
    if holdout_start <= development_end:
        return False, "holdout_overlaps_development_period"
    return True, "ok"


def independent_adoption_gate(
    development: Mapping[str, Any],
    holdout: Mapping[str, Any],
    *,
    min_holdout_rows: int = 100,
) -> dict[str, Any]:
    """Require improvement over a locked, independent OOS holdout.

    Development OOS may be used to generate/select a candidate. The holdout is
    never used for selection, calibration, threshold tuning, or other model
    decisions. Promotion requires primary LogLoss improvement, no regression in
    Brier/ECE/Accuracy, sufficient holdout rows, and explicit integrity evidence.
    """
    integrity_ok, integrity_reason = _holdout_integrity(holdout)
    if not integrity_ok:
        return {
            "status": "HOLD",
            "reason": integrity_reason,
            "oos_claimed": False,
            "promotion_authority": "deterministic_research_engine",
        }
    if int(holdout.get("n", 0)) < min_holdout_rows:
        return {"status": "HOLD", "reason": "independent_holdout_too_small", "oos_claimed": False}
    if holdout.get("same_oos") is not True:
        return {"status": "HOLD", "reason": "holdout_is_not_same_oos", "oos_claimed": False}

    base = holdout.get("baseline", {})
    cand = holdout.get("candidate", {})
    required = ("logloss", "brier", "ece", "accuracy")
    if not all(k in base and k in cand for k in required):
        return {"status": "HOLD", "reason": "incomplete_holdout_metrics", "oos_claimed": False}

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
        "holdout_integrity_verified": True,
        "promotion_authority": "deterministic_research_engine",
    }
