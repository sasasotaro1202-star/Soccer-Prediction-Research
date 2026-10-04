from __future__ import annotations

import pandas as pd
import pytest

from src.research.pit_training import filter_prior_mature_training


def test_filter_keeps_only_prior_mature_states():
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
    result = filter_prior_mature_training(
        frame, pd.Timestamp("2026-01-01T10:00:00Z")
    )
    assert result["prediction_pit_cutoff_utc"].tolist() == [
        pd.Timestamp("2026-01-01T08:00:00Z"),
    ]


def test_filter_rejects_missing_required_columns():
    with pytest.raises(RuntimeError, match="missing required timestamp"):
        filter_prior_mature_training(
            pd.DataFrame({"prediction_pit_cutoff_utc": ["2026-01-01T08:00:00Z"]}),
            pd.Timestamp("2026-01-01T10:00:00Z"),
        )


def test_filter_rejects_invalid_timestamp():
    frame = pd.DataFrame({
        "prediction_pit_cutoff_utc": ["invalid"],
        "experience_available_at_utc": ["2026-01-01T09:00:00Z"],
    })
    with pytest.raises(RuntimeError, match="missing or invalid"):
        filter_prior_mature_training(
            frame, pd.Timestamp("2026-01-01T10:00:00Z")
        )


def test_filter_rejects_impossible_own_maturity_order():
    frame = pd.DataFrame({
        "prediction_pit_cutoff_utc": ["2026-01-01T10:00:00Z"],
        "experience_available_at_utc": ["2026-01-01T09:00:00Z"],
    })
    with pytest.raises(RuntimeError, match="outcome maturity"):
        filter_prior_mature_training(
            frame, pd.Timestamp("2026-01-01T12:00:00Z")
        )


def test_filter_rejects_maturity_equal_to_own_prediction_cutoff():
    frame = pd.DataFrame({
        "prediction_pit_cutoff_utc": ["2026-01-01T10:00:00Z"],
        "experience_available_at_utc": ["2026-01-01T10:00:00Z"],
    })
    with pytest.raises(RuntimeError, match="outcome maturity"):
        filter_prior_mature_training(
            frame, pd.Timestamp("2026-01-01T12:00:00Z")
        )
