from __future__ import annotations

import pandas as pd

from src.prediction.runner import _filter_prediction_window, _promote_current_matchday_pit


def test_current_matchday_snapshot_can_close_live_pit_without_changing_historical_semantics():
    now = pd.Timestamp("2026-10-01T12:00:00Z")
    frame = pd.DataFrame(
        [{
            "match_id": "m1",
            "kickoff_utc": "2026-10-01T12:30:00Z",
            "source_available_at_utc": pd.NaT,
            "pit_verified": False,
            "matchday_available_at_utc": "2026-10-01T11:59:00Z",
            "matchday_pit_verified": True,
        }, {
            "match_id": "m2",
            "kickoff_utc": "2026-10-01T12:30:00Z",
            "source_available_at_utc": "2026-10-01T10:00:00Z",
            "pit_verified": True,
            "matchday_available_at_utc": "2026-10-01T11:59:00Z",
            "matchday_pit_verified": True,
        }],
    )
    out = _promote_current_matchday_pit(frame, now)
    assert bool(out.loc[out["match_id"].eq("m1"), "pit_verified"].iloc[0])
    assert pd.Timestamp(out.loc[out["match_id"].eq("m1"), "source_available_at_utc"].iloc[0]) == pd.Timestamp("2026-10-01T11:59:00Z")
    assert out.loc[out["match_id"].eq("m1"), "prediction_availability_basis"].iloc[0] == "CURRENT_MATCHDAY_RETRIEVAL_LOWER_BOUND"
    assert out.loc[out["match_id"].eq("m2"), "prediction_availability_basis"].iloc[0] == "SOURCE_AVAILABILITY"


def test_prediction_window_selects_only_25_to_35_minutes_before_kickoff():
    now = pd.Timestamp("2026-10-01T12:00:00Z")
    frame = pd.DataFrame([
        {"match_id": "early", "kickoff_utc": "2026-10-01T12:36:00Z"},
        {"match_id": "low", "kickoff_utc": "2026-10-01T12:35:00Z"},
        {"match_id": "center", "kickoff_utc": "2026-10-01T12:30:00Z"},
        {"match_id": "high", "kickoff_utc": "2026-10-01T12:25:00Z"},
        {"match_id": "late", "kickoff_utc": "2026-10-01T12:24:00Z"},
    ])
    selected, meta = _filter_prediction_window(frame, now, 30, 5)
    assert selected["match_id"].tolist() == ["center", "high", "low"]
    assert meta["fixtures_in_window"] == 3


def test_prediction_window_rejects_invalid_tolerance():
    frame = pd.DataFrame([{"match_id": "m", "kickoff_utc": "2026-10-01T12:30:00Z"}])
    try:
        _filter_prediction_window(
            frame,
            pd.Timestamp("2026-10-01T12:00:00Z"),
            30,
            30,
        )
    except ValueError:
        return
    raise AssertionError("invalid tolerance must fail closed")
