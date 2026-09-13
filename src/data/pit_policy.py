from __future__ import annotations

import os

import pandas as pd

# Historical result availability must not be inferred from event time in the
# production OOS gate. A result+24h assumption can be useful for diagnostics,
# but it is not source-publication evidence and therefore defaults OFF.
RESULT_AVAILABILITY_LAG = pd.Timedelta(hours=24)


def inferred_result_available_at(event_time):
    """Return a conservative diagnostic-only lower bound (event + 24h)."""
    ts = pd.to_datetime(event_time, utc=True, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    return ts + RESULT_AVAILABILITY_LAG


def result_feature_available_at(event_time):
    """Return source publication evidence only; never fabricate a timestamp.

    The legacy inferred 24h rule is available behind an explicit environment
    switch for diagnostics/tests, but production runs remain fail-closed.
    """
    allow_inferred = os.getenv("ALLOW_INFERRED_RESULT_AVAILABILITY", "false").lower() == "true"
    return inferred_result_available_at(event_time) if allow_inferred else pd.NaT


def is_available_by_cutoff(event_time, cutoff) -> bool:
    available = result_feature_available_at(event_time)
    cutoff_ts = pd.to_datetime(cutoff, utc=True, errors="coerce")
    return bool(pd.notna(available) and pd.notna(cutoff_ts) and available <= cutoff_ts)
