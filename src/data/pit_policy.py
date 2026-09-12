from __future__ import annotations

import pandas as pd

# Conservative, deterministic availability policy for result-derived historical state.
# A completed match is not allowed to influence a later prediction until this lag
# has elapsed. This avoids pretending that archive retrieval time is source time.
RESULT_AVAILABILITY_LAG = pd.Timedelta(hours=24)


def result_feature_available_at(event_time):
    ts = pd.to_datetime(event_time, utc=True, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    return ts + RESULT_AVAILABILITY_LAG


def is_available_by_cutoff(event_time, cutoff) -> bool:
    available = result_feature_available_at(event_time)
    cutoff_ts = pd.to_datetime(cutoff, utc=True, errors="coerce")
    return bool(pd.notna(available) and pd.notna(cutoff_ts) and available <= cutoff_ts)
