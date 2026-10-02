from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any


SCHEMA_VERSION = 1


def _timestamp(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nat", "none", "nan"}:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None
    return dt.astimezone(timezone.utc).isoformat()


def build_row_pit_lineage(
    *,
    match_id: Any,
    kickoff_utc: Any,
    prediction_cutoff_at_utc: Any,
    feature_source_max_available_at_utc: Any,
    pit_verified: Any,
    outcome_source_available_at_utc: Any = None,
) -> dict[str, Any]:
    """Build immutable row-level PIT lineage metadata for an OOS case.

    The target/outcome publication time is recorded separately from the
    prediction-time feature availability boundary. A verified row must expose
    parseable timestamps and satisfy feature availability <= prediction cutoff.
    """
    match = str(match_id).strip()
    kickoff = _timestamp(kickoff_utc)
    cutoff = _timestamp(prediction_cutoff_at_utc)
    feature_available = _timestamp(feature_source_max_available_at_utc)
    outcome_available = _timestamp(outcome_source_available_at_utc)
    verified = bool(pit_verified) if isinstance(pit_verified, (bool, int)) else False

    if not match:
        status = "FAIL_MISSING_MATCH_ID"
    elif kickoff is None:
        status = "FAIL_MISSING_OR_INVALID_KICKOFF"
    elif cutoff is None:
        status = "FAIL_MISSING_OR_INVALID_PREDICTION_CUTOFF"
    elif feature_available is None:
        status = "FAIL_MISSING_OR_INVALID_FEATURE_AVAILABILITY"
    elif not verified:
        status = "FAIL_PIT_NOT_VERIFIED"
    else:
        cutoff_dt = datetime.fromisoformat(cutoff)
        feature_dt = datetime.fromisoformat(feature_available)
        kickoff_dt = datetime.fromisoformat(kickoff)
        if feature_dt > cutoff_dt:
            status = "FAIL_FEATURE_AFTER_CUTOFF"
        elif kickoff_dt <= cutoff_dt:
            status = "FAIL_KICKOFF_NOT_AFTER_CUTOFF"
        else:
            status = "PASS"

    payload = {
        "schema_version": SCHEMA_VERSION,
        "match_id": match,
        "kickoff_utc": kickoff,
        "prediction_cutoff_at_utc": cutoff,
        "feature_source_max_available_at_utc": feature_available,
        "pit_verified": verified,
        "outcome_source_available_at_utc": outcome_available,
        "status": status,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        **payload,
        "lineage_hash": "pit:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24],
    }
