from __future__ import annotations

import pandas as pd

from src.data.matchday_intelligence_fetch import _required_calendar_days


def test_required_calendar_days_covers_horizon_end_date():
    now = pd.Timestamp("2026-09-27T15:39:00Z")
    assert _required_calendar_days(
        now=now,
        discovery_horizon_hours=48,
        requested_days=2,
    ) == 3


def test_required_calendar_days_respects_larger_requested_window():
    now = pd.Timestamp("2026-09-27T15:39:00Z")
    assert _required_calendar_days(
        now=now,
        discovery_horizon_hours=12,
        requested_days=4,
    ) == 4


def test_required_calendar_days_never_returns_zero():
    now = pd.Timestamp("2026-09-27T15:39:00Z")
    assert _required_calendar_days(
        now=now,
        discovery_horizon_hours=1,
        requested_days=0,
    ) == 1
