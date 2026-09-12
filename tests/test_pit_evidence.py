import pandas as pd

from src.data.pit_evidence import validate_evidence_frame


def test_verified_evidence_requires_available_at_or_before_event():
    df = pd.DataFrame([
        {"event_time_utc": "2024-01-01T12:00:00Z", "source_available_at_utc": "2024-01-01T11:00:00Z", "pit_evidence_status": "VERIFIED"},
        {"event_time_utc": "2024-01-02T12:00:00Z", "source_available_at_utc": "2024-01-02T13:00:00Z", "pit_evidence_status": "VERIFIED"},
    ])
    ok, summary = validate_evidence_frame(df)
    assert not ok
    assert summary["invalid_verified"] == 1


def test_unverifiable_rows_are_not_promoted():
    df = pd.DataFrame([
        {"event_time_utc": "2024-01-01T12:00:00Z", "source_available_at_utc": None, "pit_evidence_status": "UNVERIFIABLE"},
    ])
    ok, summary = validate_evidence_frame(df)
    assert ok
    assert summary["verified"] == 0
