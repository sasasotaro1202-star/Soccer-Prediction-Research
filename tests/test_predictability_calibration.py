from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.predictability_calibration import calibrate


def _rows(n: int = 300) -> pd.DataFrame:
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for i in range(n):
        kickoff = base + pd.Timedelta(hours=i)
        raw = 0.15 if i % 2 == 0 else 0.85
        correct = i % 2 == 0
        rows.append({
            "match_id": f"m{i}",
            "prediction_state_id": f"s{i}",
            "kickoff_utc": kickoff,
            "prediction_pit_cutoff_utc": kickoff - pd.Timedelta(hours=2),
            "experience_available_at_utc": kickoff + pd.Timedelta(hours=2),
            "prediction_pit_gate": "PASS",
            "p_home": 0.70 if correct else 0.10,
            "p_draw": 0.20,
            "p_away": 0.10 if correct else 0.70,
            "actual_result": "H" if correct else "A",
            "predictability_score": raw,
        })
    return pd.DataFrame(rows)


def test_empty_calibration_is_warmup():
    state = calibrate(pd.DataFrame())
    assert state["status"] == "WARMUP"
    assert state["production_usable"] is False


def test_calibration_is_chronological_and_research_only():
    state = calibrate(_rows(), history_rows=120, block_size=60)
    assert state["status"] in {"PROMOTION_CANDIDATE", "HOLD", "INSUFFICIENT_OOS"}
    assert state["production_usable"] is False
    assert state["safety_contract"]["production_probabilities_changed"] is False
    for block in state["oos_blocks"]:
        assert block["training_rows"] >= 120
        assert pd.Timestamp(block["oos_start"]) > pd.Timestamp("2026-01-01T00:00:00Z")


def test_invalid_probability_and_duplicate_identity_fail_closed():
    df = _rows(130)
    df.loc[1, "match_id"] = df.loc[0, "match_id"]
    df["prediction_state_id"] = pd.NA
    with pytest.raises(RuntimeError):
        calibrate(df)


def test_calibration_cases_are_bounded():
    state = calibrate(_rows(), history_rows=120, block_size=60)
    for row in state.get("oos_cases", []):
        assert 0.0 <= float(row["raw_predictability"]) <= 1.0
        assert 0.01 <= float(row["calibrated_predictability"]) <= 0.99
        assert int(row["correct"]) in {0, 1}
