import pandas as pd
import pytest
from src.core.pit import leakage_gate


def test_pit_passes_when_source_is_available_before_cutoff():
    d = pd.DataFrame([{
        "event_time_utc": "2025-01-01T18:00:00Z",
        "source_available_at_utc": "2025-01-01T18:30:00Z",
        "retrieved_at_utc": "2025-01-01T18:31:00Z",
        "prediction_cutoff_at_utc": "2025-01-02T18:00:00Z",
    }])
    assert leakage_gate(d).iloc[0].leakage_gate_status == "PASS"


def test_retrieval_time_cannot_replace_source_availability():
    d = pd.DataFrame([{
        "event_time_utc": "2025-01-01T18:00:00Z",
        "retrieved_at_utc": "2025-01-01T18:31:00Z",
        "prediction_cutoff_at_utc": "2025-01-02T18:00:00Z",
    }])
    assert leakage_gate(d).iloc[0].leakage_gate_status == "FAIL"
