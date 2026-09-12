"""Compatibility exports for the time-precise PIT archive adapter."""

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.data import pit_source_adapter_v2 as _impl
from src.data.pit_source_adapter_v2 import *
from src.data.pit_source_adapter_v2 import _result_lower_bound


def _utc(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return None
    if len(text) == 14 and text.isdigit():
        try:
            return datetime.strptime(text, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _date_key(value: Any) -> str | None:
    """Parse source dates without interpreting ISO YYYY-MM-DD as day-first."""
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return None
    parsed = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    return None if pd.isna(parsed) else parsed.date().isoformat()


# Internal functions in v2 resolve helpers in the v2 module namespace.
# Patch both parsers there so Wayback timestamps and ISO source dates use
# unambiguous calendar semantics throughout PIT replay.
_impl._utc = _utc
_impl._date_key = _date_key


def competition_adapter_matrix() -> pd.DataFrame:
    rows = []
    for competition, spec in COMPETITION_ADAPTERS.items():
        rows.append({
            "competition": competition,
            "source": spec.get("source"),
            "source_code": spec.get("source_code"),
            "adapter": spec.get("adapter"),
            "status": "IMPLEMENTED" if spec.get("adapter") else "UNVERIFIED",
        })
    return pd.DataFrame(rows)
