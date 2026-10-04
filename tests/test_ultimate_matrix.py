from __future__ import annotations

import pandas as pd
import pytest

from src.research.ultimate_matrix import (
    CALIBRATION_MODES,
    MODEL_ECOLOGY,
    ROUTING_MODES,
    TRAINING_WINDOWS,
    summarize_oos,
    _validate_input,
)
from src.research.feature_set_variants import variant_catalog


def test_ultimate_catalog_has_broad_feature_space():
    assert len(variant_catalog()) >= 20


def test_ultimate_matrix_axes_are_broad_and_explicit():
    assert len(variant_catalog()) >= 30
    assert len(MODEL_ECOLOGY) >= 30
    assert len(TRAINING_WINDOWS) >= 5
    assert set(CALIBRATION_MODES) == {"none", "global", "context", "full"}
    assert set(ROUTING_MODES) == {"global", "context", "dynamic"}


def test_summarize_oos_has_disjoint_screen_model_config_and_locked_layers():
    wf = pd.DataFrame(
        {
            "n": [100] * 8,
            "logloss": [1.00, 0.90, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55],
            "brier": [0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23],
            "accuracy": [0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57],
            "ece": [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03],
            "oos_window_signature": ["sig"] * 8,
        }
    )
    out = summarize_oos(wf)
    assert out["screen_blocks"] == 2
    assert out["model_confirm_blocks"] == 2
    assert out["config_confirm_blocks"] == 2
    assert out["locked_blocks"] == 2
    assert out["screen_n"] == 200
    assert out["model_confirm_n"] == 200
    assert out["config_confirm_n"] == 200
    assert out["locked_n"] == 200
    assert out["screen_logloss"] == pytest.approx(0.95)
    assert out["model_confirm_logloss"] == pytest.approx(0.775)
    assert out["config_confirm_logloss"] == pytest.approx(0.675)
    assert out["locked_logloss"] == pytest.approx(0.575)


def test_summarize_oos_rejects_too_few_blocks():
    wf = pd.DataFrame(
        {
            "n": [100] * 7,
            "logloss": [1.0] * 7,
            "brier": [0.3] * 7,
            "accuracy": [0.5] * 7,
            "ece": [0.1] * 7,
            "oos_window_signature": ["sig"] * 7,
        }
    )
    with pytest.raises(RuntimeError, match="at least 8 chronological OOS blocks"):
        summarize_oos(wf)


def test_validate_input_rejects_future_feature_availability():
    frame = pd.DataFrame(
        {
            "match_id": ["m1"],
            "kickoff_utc": ["2026-01-01T10:00:00Z"],
            "target": [0],
            "pit_verified": [True],
            "prediction_cutoff_at_utc": ["2026-01-01T09:00:00Z"],
            "feature_source_max_available_at_utc": ["2026-01-01T09:30:00Z"],
        }
    )
    path = "/tmp/ultimate_matrix_bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="PIT validation failed"):
        _validate_input(path)
