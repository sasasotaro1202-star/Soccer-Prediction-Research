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
