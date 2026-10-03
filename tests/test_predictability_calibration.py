from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.research.predictability_calibration import calibrate


def _rows(n: int = 300) -> pd.DataFrame:
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for i in range(n):
        kickoff = base + pd.to_timedelta(i, unit="h")
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
            "predictive_entropy": 1.0 - raw,
            "model_disagreement": 1.0 - raw,
            "covariate_drift": 1.0 - raw,
            "history_support_risk": 1.0 - raw,
            "routing_risk": 1.0 - raw,
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


def test_calibration_derives_raw_predictability_from_shadow_telemetry():
    df = _rows(300)
    telemetry = (
        "predictive_entropy",
        "model_disagreement",
        "covariate_drift",
        "history_support_risk",
        "routing_risk",
    )
    df = df.drop(columns=list(telemetry))
    for name in telemetry:
        df["shadow_" + name] = 0.2
    # Keep both correct and incorrect matured labels so the binary
    # calibrator has a valid two-class training target.
    df.loc[1, "actual_result"] = "H"
    df.loc[3, "actual_result"] = "D"
    state = calibrate(df, history_rows=120, block_size=60)
    assert state["rows"] == 300
    assert state["production_usable"] is False
    assert state["oos_blocks"]
    assert state["safety_contract"]["calibration_training_is_prior_history_only"] is True


def test_calibration_always_exposes_frozen_holdout_safety_key():
    state = calibrate(pd.DataFrame())
    assert state["safety_contract"]["frozen_holdout_touched"] is False



def test_predictability_workflow_warmup_uses_frozen_holdout_key():
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "soccer-predictability-research.yml"
    text = workflow.read_text(encoding="utf-8")
    assert '"frozen_holdout_touched": False' in text
    assert '"locked_holdout_touched": False' not in text


def test_calibration_gate_requires_meaningful_stable_development():
    from src.research import predictability_calibration as mod

    development = pd.DataFrame({
        "calibrated_logloss": [0.4, 0.5, 0.7],
        "raw_logloss": [0.5, 0.6, 0.65],
    })
    assert mod._development_improvement_rate(development) == 2 / 3

    stronger = pd.DataFrame({
        "calibrated_logloss": [0.485, 0.582, 0.63],
        "raw_logloss": [0.5, 0.6, 0.65],
    })
    assert mod._development_improvement_rate(stronger) == 1.0


def test_calibration_oos_excludes_immature_prior_outcomes_from_training():
    from src.research.predictability_calibration import calibrate

    base = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for i in range(360):
        kickoff = base + pd.to_timedelta(i, unit="6h")
        correct = i % 2 == 0
        raw = 0.15 if correct else 0.85
        rows.append({
            "match_id": f"pit-m{i}",
            "prediction_state_id": f"pit-s{i}",
            "kickoff_utc": kickoff,
            "prediction_pit_cutoff_utc": kickoff - pd.Timedelta(hours=2),
            "experience_available_at_utc": kickoff + pd.Timedelta(hours=2),
            "prediction_pit_gate": "PASS",
            "p_home": 0.70 if correct else 0.10,
            "p_draw": 0.20,
            "p_away": 0.10 if correct else 0.70,
            "actual_result": "H" if correct else "A",
            "predictive_entropy": 1.0 - raw,
            "model_disagreement": 1.0 - raw,
            "covariate_drift": 1.0 - raw,
            "history_support_risk": 1.0 - raw,
            "routing_risk": 1.0 - raw,
        })
    frame = pd.DataFrame(rows)
    frame.loc[119, "experience_available_at_utc"] = (
        frame.loc[120, "prediction_pit_cutoff_utc"] + pd.Timedelta(hours=1)
    )
    state = calibrate(frame, history_rows=120, block_size=60)
    assert state["oos_blocks"]
    assert state["oos_blocks"][0]["block"] == 1
    assert state["oos_blocks"][0]["training_rows"] == 179
    assert state["oos_blocks"][0]["raw_training_rows"] == 180
    assert state["oos_blocks"][0]["excluded_immature_training_rows"] == 1


def test_maturity_training_helper_excludes_future_outcomes_and_fails_closed():
    from src.research.pit_training import filter_prior_mature_training

    frame = pd.DataFrame({
        "prediction_pit_cutoff_utc": [
            "2026-01-01T08:00:00Z",
            "2026-01-01T09:00:00Z",
            "2026-01-01T10:00:00Z",
        ],
        "experience_available_at_utc": [
            "2026-01-01T09:00:00Z",
            "2026-01-01T11:00:00Z",
            "2026-01-01T10:00:00Z",
        ],
    })
    eligible = filter_prior_mature_training(
        frame, pd.Timestamp("2026-01-01T10:00:00Z")
    )
    assert len(eligible) == 2
    assert eligible["prediction_pit_cutoff_utc"].tolist() == [
        "2026-01-01T08:00:00Z",
        "2026-01-01T09:00:00Z",
    ]


def test_maturity_training_helper_missing_timestamp_fails_closed():
    from src.research.pit_training import filter_prior_mature_training

    frame = pd.DataFrame({
        "prediction_pit_cutoff_utc": ["2026-01-01T08:00:00Z"],
        "experience_available_at_utc": [pd.NaT],
    })
    with pytest.raises(RuntimeError, match="maturity"):
        filter_prior_mature_training(
            frame, pd.Timestamp("2026-01-01T10:00:00Z")
        )
