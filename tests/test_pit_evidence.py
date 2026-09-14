import pandas as pd

from src.data.pit_evidence import validate_evidence_frame
from src.data.completion_gate import _merge_pit_evidence


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


def test_pit_evidence_merge_keeps_timestamp_dtype_and_exact_values():
    history = pd.DataFrame({"source_available_at_utc": pd.to_datetime([None, None], utc=True)})
    enriched = pd.DataFrame({
        "source_available_at_utc": ["2024-01-01T13:00:00Z", None],
        "pit_evidence_status": ["VERIFIED", "UNVERIFIABLE"],
        "pit_evidence_reason": ["archive", "missing"],
    }, index=history.index)
    merged = _merge_pit_evidence(history, enriched)
    assert str(merged["source_available_at_utc"].dtype) == "datetime64[ns, UTC]"
    assert merged.loc[0, "source_available_at_utc"] == pd.Timestamp("2024-01-01T13:00:00Z")
    assert pd.isna(merged.loc[1, "source_available_at_utc"])
    assert merged.loc[0, "pit_evidence_status"] == "VERIFIED"


def test_pit_evidence_merge_never_turns_unknown_timestamp_into_verified():
    history = pd.DataFrame({"source_available_at_utc": pd.to_datetime([None], utc=True)})
    enriched = pd.DataFrame({
        "source_available_at_utc": [None],
        "pit_evidence_status": ["UNVERIFIABLE"],
    }, index=history.index)
    merged = _merge_pit_evidence(history, enriched)
    assert pd.isna(merged.loc[0, "source_available_at_utc"])
    assert merged.loc[0, "pit_evidence_status"] == "UNVERIFIABLE"
