from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.walk_forward import run_walk_forward


def _frame(**overrides) -> pd.DataFrame:
    row = {
        "match_id": "m1",
        "kickoff_utc": "2026-01-01T10:00:00Z",
        "prediction_cutoff_at_utc": "2026-01-01T09:00:00Z",
        "feature_source_max_available_at_utc": "2026-01-01T08:00:00Z",
        "pit_verified": True,
        "target": 0,
        "elo_diff": 50.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_walk_forward_requires_explicit_pit_provenance_columns():
    frame = _frame().drop(columns=["feature_source_max_available_at_utc"])
    with pytest.raises(RuntimeError, match="explicit PIT provenance columns"):
        run_walk_forward(frame, ["elo_diff"], min_train=1, oos_block=1)


def test_walk_forward_rejects_post_cutoff_feature_availability():
    frame = _frame(
        feature_source_max_available_at_utc="2026-01-01T09:30:00Z",
    )
    with pytest.raises(RuntimeError, match="PIT provenance validation failed"):
        run_walk_forward(frame, ["elo_diff"], min_train=1, oos_block=1)


def test_walk_forward_rejects_cutoff_after_kickoff():
    frame = _frame(
        prediction_cutoff_at_utc="2026-01-01T10:30:00Z",
    )
    with pytest.raises(RuntimeError, match="PIT provenance validation failed"):
        run_walk_forward(frame, ["elo_diff"], min_train=1, oos_block=1)
