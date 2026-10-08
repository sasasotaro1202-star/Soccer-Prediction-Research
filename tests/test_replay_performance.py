from __future__ import annotations

import pandas as pd
import pytest

from src.research.replay_performance import validate_replay_input


def _safe_rows(n: int = 3000) -> pd.DataFrame:
    base = pd.Timestamp("2020-01-01T12:00:00Z")
    rows = []
    for i in range(n):
        cutoff = base + pd.Timedelta(minutes=i)
        rows.append({
            "match_id": str(i),
            "kickoff_utc": cutoff + pd.Timedelta(minutes=60),
            "prediction_cutoff_at_utc": cutoff,
            "feature_source_max_available_at_utc": cutoff - pd.Timedelta(minutes=5),
            "source_available_at_utc": cutoff + pd.Timedelta(minutes=180),
            "pit_verified": True,
            "home_goals": 1,
            "away_goals": 0,
            "target": 0,
        })
    return pd.DataFrame(rows)


def test_validate_replay_input_accepts_pit_safe_rows(tmp_path):
    p = tmp_path / "replay.csv"
    _safe_rows().to_csv(p, index=False)
    frame, report = validate_replay_input(p)
    assert len(frame) == 3000
    assert report["status"] == "PASS"
    assert report["pit_verified_rows"] == 3000


def test_validate_replay_input_rejects_feature_leakage(tmp_path):
    p = tmp_path / "replay.csv"
    df = _safe_rows()
    df.loc[0, "feature_source_max_available_at_utc"] = df.loc[0, "prediction_cutoff_at_utc"] + pd.Timedelta(minutes=1)
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="feature availability after prediction cutoff"):
        validate_replay_input(p)


def test_validate_replay_input_excludes_unverified_rows(tmp_path):
    p = tmp_path / "replay.csv"
    df = _safe_rows(3001)
    df.loc[3000, "pit_verified"] = False
    df.loc[3000, "prediction_cutoff_at_utc"] = pd.NaT
    df.loc[3000, "feature_source_max_available_at_utc"] = pd.NaT
    df.to_csv(p, index=False)
    frame, report = validate_replay_input(p)
    assert len(frame) == 3000
    assert report["input_rows"] == 3001
    assert report["pit_verified_rows"] == 3000
    assert report["non_verified_rows_excluded"] == 1



def test_validate_replay_input_rejects_outcome_availability_before_kickoff(tmp_path):
    p = tmp_path / "replay.csv"
    df = _safe_rows()
    df.loc[0, "source_available_at_utc"] = df.loc[0, "kickoff_utc"] - pd.Timedelta(minutes=1)
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="outcome availability before kickoff"):
        validate_replay_input(p)


def test_validate_replay_input_rejects_missing_outcome_availability(tmp_path):
    p = tmp_path / "replay.csv"
    df = _safe_rows()
    df.loc[0, "source_available_at_utc"] = pd.NaT
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing outcome availability timestamps"):
        validate_replay_input(p)
