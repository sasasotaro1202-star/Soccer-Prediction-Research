from __future__ import annotations

"""PIT provenance policy for public sources.

A row is production-eligible only when its availability timestamp is demonstrably
at or before the prediction cutoff. Retrieval time alone is never sufficient.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PITDecision:
    status: str
    reason: str
    available_at_utc: pd.Timestamp | pd.NaT
    evidence_type: str


def _ts(value: Any):
    if value is None or value == "" or pd.isna(value):
        return pd.NaT
    try:
        t = pd.Timestamp(value)
    except Exception:
        return pd.NaT
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("UTC")


def decide_pit(
    *,
    event_time: Any,
    prediction_cutoff: Any,
    source_available_at: Any,
    evidence_type: str,
    evidence_strength: str = "UNVERIFIED",
) -> PITDecision:
    event = _ts(event_time)
    cutoff = _ts(prediction_cutoff)
    available = _ts(source_available_at)

    if pd.isna(event):
        return PITDecision("REJECT", "missing_event_time", pd.NaT, evidence_type)
    if pd.isna(cutoff):
        return PITDecision("REJECT", "missing_prediction_cutoff", pd.NaT, evidence_type)
    if event <= cutoff:
        return PITDecision("REJECT", "event_not_after_prediction_cutoff", available, evidence_type)
    if pd.isna(available):
        return PITDecision("REJECT", "availability_time_unproven", pd.NaT, evidence_type)
    if available > cutoff:
        return PITDecision("REJECT", "source_available_after_cutoff", available, evidence_type)
    if evidence_strength != "VERIFIED":
        return PITDecision("REJECT", "evidence_not_verified", available, evidence_type)
    return PITDecision("VERIFIED", "point_in_time_proven", available, evidence_type)


def apply_pit_gate(frame: pd.DataFrame, prediction_cutoff_col: str = "prediction_cutoff_utc") -> pd.DataFrame:
    out = frame.copy()
    required = {"kickoff_utc", "source_available_at_utc", prediction_cutoff_col}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"missing PIT columns: {sorted(missing)}")

    statuses = []
    reasons = []
    eligible = []
    for _, row in out.iterrows():
        d = decide_pit(
            event_time=row["kickoff_utc"],
            prediction_cutoff=row[prediction_cutoff_col],
            source_available_at=row["source_available_at_utc"],
            evidence_type=str(row.get("pit_evidence_type", "UNKNOWN")),
            evidence_strength=str(row.get("pit_evidence_strength", "UNVERIFIED")),
        )
        statuses.append(d.status)
        reasons.append(d.reason)
        eligible.append(d.status == "VERIFIED")

    out["pit_gate_status"] = statuses
    out["pit_gate_reason"] = reasons
    out["production_eligible"] = eligible
    return out
