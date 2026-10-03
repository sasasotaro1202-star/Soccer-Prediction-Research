from __future__ import annotations

import pandas as pd

from src.research.predictability_calibration import calibrate


def _base_rows(n: int = 180) -> pd.DataFrame:
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for i in range(n):
        kickoff = base + pd.to_timedelta(i, unit="h")
        rows.append({
            "match_id": f"pit-{i}",
            "prediction_state_id": f"pit-state-{i}",
            "kickoff_utc": kickoff,
            "prediction_pit_cutoff_utc": kickoff - pd.Timedelta(hours=2),
            "experience_available_at_utc": kickoff + pd.Timedelta(hours=2),
            "prediction_pit_gate": "PASS",
            "p_home": 0.50,
            "p_draw": 0.20,
            "p_away": 0.30,
            "predictive_entropy": 0.2,
            "model_disagreement": 0.2,
            "covariate_drift": 0.2,
            "history_support_risk": 0.2,
            "routing_risk": 0.2,
            "actual_result": "H" if i % 2 == 0 else "A",
        })
    return pd.DataFrame(rows)


def test_rejects_post_cutoff_source_availability():
    df = _base_rows()
    df["source_available_at_utc"] = (
        df["prediction_pit_cutoff_utc"] + pd.Timedelta(minutes=1)
    )
    state = calibrate(df, history_rows=120, block_size=60)
    assert state["rows"] == 0
    assert state["status"] == "INSUFFICIENT_OOS"


def test_rejects_post_cutoff_retrieval():
    df = _base_rows()
    df["retrieved_at_utc"] = (
        df["prediction_pit_cutoff_utc"] + pd.Timedelta(minutes=1)
    )
    state = calibrate(df, history_rows=120, block_size=60)
    assert state["rows"] == 0
    assert state["status"] == "INSUFFICIENT_OOS"


def test_rejects_retrieval_before_source_availability():
    df = _base_rows()
    df["source_available_at_utc"] = (
        df["prediction_pit_cutoff_utc"] - pd.Timedelta(minutes=5)
    )
    df["retrieved_at_utc"] = (
        df["prediction_pit_cutoff_utc"] - pd.Timedelta(minutes=10)
    )
    state = calibrate(df, history_rows=120, block_size=60)
    assert state["rows"] == 0
    assert state["status"] == "INSUFFICIENT_OOS"
