from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.temporal_conformal_sets import temporal_prediction_sets


def _rows(n: int = 100) -> pd.DataFrame:
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for i in range(n):
        kickoff = base + pd.Timedelta(hours=i)
        rows.append({
            "match_id": f"m{i}",
            "prediction_state_id": f"s{i}",
            "kickoff_utc": kickoff,
            "prediction_pit_cutoff_utc": kickoff - pd.Timedelta(hours=2),
            "experience_available_at_utc": kickoff + pd.Timedelta(hours=2),
            "prediction_pit_gate": "PASS",
            "p_home": 0.80 if i % 3 == 0 else 0.10 if i % 3 == 1 else 0.05,
            "p_draw": 0.15 if i % 3 == 0 else 0.80 if i % 3 == 1 else 0.10,
            "p_away": 0.05 if i % 3 == 0 else 0.10 if i % 3 == 1 else 0.85,
            "actual_result": "H" if i % 3 == 0 else "D" if i % 3 == 1 else "A",
        })
    return pd.DataFrame(rows)


def test_empty_conformal_ledger_is_warmup():
    state = temporal_prediction_sets(pd.DataFrame())
    assert state["status"] == "WARMUP"
    assert state["eligible_rows"] == 0
    assert state["production_usable"] is False


def test_maturity_is_post_kickoff():
    df = _rows(80)
    df.loc[0, "experience_available_at_utc"] = df.loc[0, "kickoff_utc"] - pd.Timedelta(minutes=1)
    with pytest.raises(RuntimeError, match="maturity"):
        temporal_prediction_sets(df)


def test_conformal_calibration_excludes_current_and_immature_outcomes():
    df = _rows(80)
    df.loc[0, "experience_available_at_utc"] = pd.Timestamp("2026-01-04T00:00:00Z")
    state = temporal_prediction_sets(df, min_calibration=10, max_calibration=20)
    rows = pd.DataFrame(state["prediction_set_rows"])
    assert int(rows.loc[10, "calibration_rows"]) == 9
    assert int(rows.loc[11, "calibration_rows"]) == 10
    assert int(rows["calibration_rows"].max()) <= 20


def test_conformal_is_research_only_and_reports_metrics():
    state = temporal_prediction_sets(_rows(100), min_calibration=10, max_calibration=30)
    assert state["status"] == "READY"
    assert state["production_usable"] is False
    assert state["safety_contract"]["production_probabilities_changed"] is False
    assert state["safety_contract"]["outcome_data_used_only_after_maturity"] is True
    assert 0.0 <= float(state["metrics"]["coverage"]) <= 1.0
    assert np.isfinite(float(state["metrics"]["mean_set_size"]))


def test_duplicate_fixture_without_state_identity_fails_closed():
    df = _rows(70)
    df.loc[1, "match_id"] = df.loc[0, "match_id"]
    df["prediction_state_id"] = pd.NA
    with pytest.raises(RuntimeError, match="duplicate match_id"):
        temporal_prediction_sets(df)
