from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class EvidenceResult:
    status: str
    source: str | None
    source_available_at_utc: str | None
    evidence_url: str | None
    reason: str


def _utc(value):
    if value is None or value == "":
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(ts) else ts


def validate_evidence_frame(df: pd.DataFrame) -> tuple[bool, dict]:
    """Validate PIT evidence independently of any particular archive provider."""
    required = {"source_available_at_utc", "event_time_utc", "pit_evidence_status"}
    missing = sorted(required - set(df.columns))
    if missing:
        return False, {"reason": "missing_columns", "columns": missing}

    event = pd.to_datetime(df["event_time_utc"], utc=True, errors="coerce")
    available = pd.to_datetime(df["source_available_at_utc"], utc=True, errors="coerce")
    verified = df["pit_evidence_status"].astype(str).eq("VERIFIED")

    invalid_verified = verified & (event.isna() | available.isna() | (available > event))
    invalid_status = ~df["pit_evidence_status"].astype(str).isin(["VERIFIED", "UNVERIFIABLE"])
    ok = not bool(invalid_verified.any() or invalid_status.any())
    return ok, {
        "rows": int(len(df)),
        "verified": int(verified.sum()),
        "invalid_verified": int(invalid_verified.sum()),
        "invalid_status": int(invalid_status.sum()),
    }


def choose_verified_evidence(candidates: Iterable[EvidenceResult]) -> EvidenceResult | None:
    """Choose only independently verifiable evidence; never promote missing evidence."""
    valid = [c for c in candidates if c.status == "VERIFIED" and c.source_available_at_utc]
    if not valid:
        return None
    valid.sort(key=lambda c: _utc(c.source_available_at_utc))
    return valid[0]
