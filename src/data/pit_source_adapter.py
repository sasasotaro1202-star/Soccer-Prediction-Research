"""Compatibility exports for the time-precise PIT archive adapter.

The Wayback CDX API returns capture timestamps as compact 14-digit strings
(`YYYYMMDDhhmmss`). The v2 adapter's timestamp parser must understand that
format; this compatibility layer patches the shared v2 parser so all internal
PIT comparisons use the same UTC interpretation.
"""

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

    # Wayback CDX capture timestamp: YYYYMMDDhhmmss.
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


# Internal functions in pit_source_adapter_v2 resolve _utc in their own
# module namespace. Patch that single parser rather than duplicating the
# adapter implementation in this compatibility module.
_impl._utc = _utc
