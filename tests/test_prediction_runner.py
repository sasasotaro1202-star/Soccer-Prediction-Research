from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.prediction.runner import _eligible_fixtures


def _fixtures(tmp_path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "ok",
                "kickoff_utc": "2026-09-15T18:00:00Z",
                "home_team": "A",
                "away_team": "B",
                "competition": "EPL",
                "source_available_at_utc": "2026-09-14T00:00:00Z",
                "pit_verified": True,
                "starter_status": "ANNOUNCED",
            },
            {
                "match_id": "late-info",
                "kickoff_utc": "2026-09-15T19:00:00Z",
                "home_team": "C",
                "away_team": "D",
                "competition": "EPL",
                "source_available_at_utc": "2026-09-15T19:30:00Z",
                "pit_verified": True,
                "starter_status": "ANNOUNCED",
            },
            {
                "match_id": "no-starters",
                "kickoff_utc": "2026-09-15T20:00:00Z",
                "home_team": "E",
                "away_team": "F",
                "competition": "EPL",
                "source_available_at_utc": "2026-09-14T00:00:00Z",
                "pit_verified": True,
                "starter_status": "EXPECTED",
            },
            {
                "match_id": "unverified",
                "kickoff_utc": "2026-09-15T21:00:00Z",
                "home_team": "G",
                "away_team": "H",
                "competition": "EPL",
                "source_available_at_utc": "2026-09-14T00:00:00Z",
                "pit_verified": False,
                "starter_status": "CONFIRMED",
            },
        ]
    )


def test_eligibility_is_forward_looking_and_starter_gated():
    d = _fixtures(Path("."))
    out = _eligible_fixtures(d, pd.Timestamp("2026-09-14T12:00:00Z"))
    assert out["match_id"].tolist() == ["ok"]


def test_missing_required_fixture_field_fails_closed():
    d = _fixtures(Path(".")).drop(columns=["source_available_at_utc"])
    with pytest.raises(RuntimeError, match="missing required columns"):
        _eligible_fixtures(d, pd.Timestamp("2026-09-14T12:00:00Z"))


def test_default_model_policy_prefers_best_available():
    import inspect
    from src.prediction.runner import run

    assert inspect.signature(run).parameters["model_policy"].default == "best_available"
