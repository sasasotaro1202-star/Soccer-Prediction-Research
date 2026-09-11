from __future__ import annotations

"""Fail-closed source-availability and PIT utilities.

A retrieved_at timestamp describes our observation of a source, not when the
source made the information public. Production/research features must carry an
independently auditable source_available_at timestamp.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def as_utc(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class PITDecision:
    allowed: bool
    reason: str


def pit_decision(
    *,
    source_available_at: Any,
    event_time: Any,
    prediction_cutoff_at: Any,
    require_source_timestamp: bool = True,
) -> PITDecision:
    """Return whether a record is admissible at a prediction cutoff.

    The event itself must precede the cutoff and the source must have made the
    information available by the cutoff. Missing source availability fails
    closed when `require_source_timestamp` is true.
    """
    cutoff = as_utc(prediction_cutoff_at)
    event = as_utc(event_time)
    available = as_utc(source_available_at)
    if cutoff is None:
        return PITDecision(False, "missing_prediction_cutoff")
    if event is None:
        return PITDecision(False, "missing_event_time")
    if event >= cutoff:
        return PITDecision(False, "event_not_before_cutoff")
    if available is None and require_source_timestamp:
        return PITDecision(False, "missing_source_available_at")
    if available is not None and available > cutoff:
        return PITDecision(False, "source_available_after_cutoff")
    return PITDecision(True, "pit_ok")


def add_pit_columns(record: dict[str, Any], *, prediction_cutoff_at: Any) -> dict[str, Any]:
    """Annotate a canonical record without inventing source timestamps."""
    out = dict(record)
    out["prediction_cutoff_at_utc"] = as_utc(prediction_cutoff_at)
    decision = pit_decision(
        source_available_at=out.get("source_available_at_utc"),
        event_time=out.get("event_time_utc"),
        prediction_cutoff_at=out.get("prediction_cutoff_at_utc"),
    )
    out["pit_allowed"] = decision.allowed
    out["pit_reason"] = decision.reason
    return out
