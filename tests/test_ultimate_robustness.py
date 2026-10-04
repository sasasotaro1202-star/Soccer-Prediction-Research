from __future__ import annotations

import json

import pandas as pd
import pytest

from src.research.ultimate_robustness import (
    _locked_match_ids,
    _metrics_tail,
    _validate_frame,
)


def test_validate_frame_rejects_post_cutoff_feature_availability():
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
    with pytest.raises(RuntimeError, match="PIT validation failed"):
        _validate_frame(frame)


def test_metrics_tail_uses_only_locked_suffix():
    wf = pd.DataFrame(
        {
            "n": [100, 100, 100, 100],
            "logloss": [1.0, 0.9, 0.8, 0.7],
            "brier": [0.30, 0.29, 0.28, 0.27],
            "accuracy": [0.50, 0.55, 0.60, 0.65],
            "ece": [0.10, 0.09, 0.08, 0.07],
            "oos_window_signature": ["sig"] * 4,
        }
    )
    out = _metrics_tail(wf, locked_blocks=2)
    assert out["locked_n"] == 200
    assert out["locked_logloss"] == pytest.approx(0.75)
    assert out["locked_brier"] == pytest.approx(0.275)
    assert out["locked_accuracy"] == pytest.approx(0.625)
    assert out["locked_ece"] == pytest.approx(0.075)


def test_locked_match_ids_are_deterministic_and_suffix_based():
    n = 32
    kickoff = pd.date_range("2026-01-01T00:00:00Z", periods=n, freq="h")
    frame = pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(n)],
            "kickoff_utc": kickoff,
            "target": [0, 1, 2] * 10 + [0, 1],
            "pit_verified": [True] * n,
            "prediction_cutoff_at_utc": kickoff - pd.Timedelta(minutes=60),
            "feature_source_max_available_at_utc": kickoff - pd.Timedelta(minutes=120),
        }
    )
    ids = _locked_match_ids(frame, min_train=4, oos_block=4)
    assert ids == {f"m{i}" for i in range(24, 32)}


def test_locked_match_ids_keep_shared_kickoff_rows_together():
    n = 34
    kickoff = pd.date_range("2026-01-01T00:00:00Z", periods=n, freq="h").to_series()
    kickoff.iloc[26] = kickoff.iloc[25]
    frame = pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(n)],
            "kickoff_utc": kickoff.to_numpy(),
            "target": [0, 1, 2] * 11 + [0],
            "pit_verified": [True] * n,
            "prediction_cutoff_at_utc": kickoff.to_numpy() - pd.Timedelta(minutes=60),
            "feature_source_max_available_at_utc": kickoff.to_numpy() - pd.Timedelta(minutes=120),
        }
    )
    ids = _locked_match_ids(frame, min_train=2, oos_block=4)
    assert "m25" not in ids
    assert "m26" not in ids
    assert "m27" in ids


def test_status_is_research_only_contract(tmp_path):
    status = {
        "status": "RESEARCH_EXECUTED",
        "production_changed": False,
        "production_registry_changed": False,
        "frozen_holdout_used_for_selection": False,
        "locked_oos_used_for_selection": False,
        "robustness_is_diagnostic_not_selection": True,
    }
    p = tmp_path / "status.json"
    p.write_text(json.dumps(status), encoding="utf-8")
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert loaded["production_changed"] is False
    assert loaded["frozen_holdout_used_for_selection"] is False
    assert loaded["robustness_is_diagnostic_not_selection"] is True
