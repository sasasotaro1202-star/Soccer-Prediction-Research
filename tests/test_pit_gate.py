import pandas as pd
import pytest

from src.data.pit_gate import apply_pit_gate, decide_pit


def test_pit_gate_accepts_verified_pre_cutoff_source():
    d = decide_pit(
        event_time="2026-09-20T18:00:00Z",
        prediction_cutoff="2026-09-20T12:00:00Z",
        source_available_at="2026-09-20T10:00:00Z",
        evidence_type="VERSIONED_SNAPSHOT",
        evidence_strength="VERIFIED",
    )
    assert d.status == "VERIFIED"


def test_pit_gate_accepts_verified_source_exactly_at_cutoff():
    d = decide_pit(
        event_time="2026-09-20T18:00:00Z",
        prediction_cutoff="2026-09-20T12:00:00Z",
        source_available_at="2026-09-20T12:00:00Z",
        evidence_type="VERSIONED_SNAPSHOT",
        evidence_strength="VERIFIED",
    )
    assert d.status == "VERIFIED"
    assert d.reason == "point_in_time_proven"


def test_pit_gate_rejects_event_at_cutoff():
    d = decide_pit(
        event_time="2026-09-20T12:00:00Z",
        prediction_cutoff="2026-09-20T12:00:00Z",
        source_available_at="2026-09-20T10:00:00Z",
        evidence_type="VERSIONED_SNAPSHOT",
        evidence_strength="VERIFIED",
    )
    assert d.status == "REJECT"
    assert d.reason == "event_not_after_prediction_cutoff"


@pytest.mark.parametrize("available", [
    None,
    "2026-09-20T12:00:01Z",
])
def test_pit_gate_rejects_unproven_or_late_source(available):
    d = decide_pit(
        event_time="2026-09-20T18:00:00Z",
        prediction_cutoff="2026-09-20T12:00:00Z",
        source_available_at=available,
        evidence_type="UNKNOWN",
        evidence_strength="UNVERIFIED",
    )
    assert d.status == "REJECT"


def test_apply_pit_gate_is_fail_closed():
    frame = pd.DataFrame([{
        "kickoff_utc": "2026-09-20T18:00:00Z",
        "source_available_at_utc": None,
        "prediction_cutoff_utc": "2026-09-20T12:00:00Z",
    }])
    out = apply_pit_gate(frame)
    assert bool(out.loc[0, "production_eligible"]) is False
    assert out.loc[0, "pit_gate_status"] == "REJECT"
