from datetime import datetime, timezone

from src.core.source_availability import pit_decision


def dt(hour: int) -> datetime:
    return datetime(2025, 1, 1, hour, 0, tzinfo=timezone.utc)


def test_missing_source_timestamp_fails_closed():
    result = pit_decision(
        source_available_at=None,
        event_time=dt(10),
        prediction_cutoff_at=dt(11),
    )
    assert not result.allowed
    assert result.reason == "missing_source_available_at"


def test_source_after_cutoff_is_rejected():
    result = pit_decision(
        source_available_at=dt(12),
        event_time=dt(10),
        prediction_cutoff_at=dt(11),
    )
    assert not result.allowed
    assert result.reason == "source_available_after_cutoff"


def test_valid_record_is_allowed():
    result = pit_decision(
        source_available_at=dt(9),
        event_time=dt(10),
        prediction_cutoff_at=dt(11),
    )
    assert result.allowed
    assert result.reason == "pit_ok"
