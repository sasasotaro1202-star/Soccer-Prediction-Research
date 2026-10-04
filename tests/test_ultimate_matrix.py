from __future__ import annotations

import pandas as pd
import pytest

from src.research.ultimate_matrix import summarize_oos, _validate_input
from src.research.feature_set_variants import variant_catalog


def test_ultimate_catalog_has_broad_feature_space():
    assert len(variant_catalog()) >= 20


def test_summarize_oos_separates_screen_confirm_and_locked():
    wf = pd.DataFrame(
        {
            "n": [100, 100, 100, 100, 100, 100],
            "logloss": [1.0, 0.9, 0.8, 0.75, 0.7, 0.65],
            "brier": [0.30, 0.29, 0.28, 0.27, 0.26, 0.25],
            "accuracy": [0.50, 0.51, 0.52, 0.53, 0.54, 0.55],
            "ece": [0.10, 0.09, 0.08, 0.07, 0.06, 0.05],
            "oos_window_signature": ["sig"] * 6,
        }
    )
    out = summarize_oos(wf)
    assert out["screen_blocks"] == 2
    assert out["confirm_blocks"] == 2
    assert out["locked_blocks"] == 2
    assert out["screen_n"] == 200
    assert out["confirm_n"] == 200
    assert out["locked_n"] == 200
    assert out["screen_logloss"] == pytest.approx(0.95)
    assert out["confirm_logloss"] == pytest.approx(0.775)
    assert out["locked_logloss"] == pytest.approx(0.675)


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
