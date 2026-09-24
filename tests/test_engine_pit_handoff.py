import pandas as pd
import pytest

from src.research.engine import _load_preflight_pit_features


def test_load_preflight_pit_features_reuses_gate_features_and_attaches_outcomes(tmp_path):
    pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "kickoff_utc": ["2024-01-01T12:00:00Z", "2024-01-02T12:00:00Z"],
            "prediction_cutoff_at_utc": ["2024-01-01T11:00:00Z", "2024-01-02T11:00:00Z"],
            "pit_verified": [True, False],
            "elo_diff": [10.0, -5.0],
        }
    ).to_csv(tmp_path / "pit_replay_features.csv", index=False)

    history = pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "home_goals": [2, 0],
            "away_goals": [1, 1],
        }
    )

    result = _load_preflight_pit_features(tmp_path, history)
    assert result is not None
    assert result["home_goals"].tolist() == [2, 0]
    assert result["away_goals"].tolist() == [1, 1]
    assert result["pit_verified"].tolist() == [True, False]
    assert result["target"].tolist() == [0, 1]


def test_load_preflight_pit_features_rejects_missing_outcome_identity(tmp_path):
    pd.DataFrame(
        {
            "match_id": ["m1"],
            "kickoff_utc": ["2024-01-01T12:00:00Z"],
            "prediction_cutoff_at_utc": ["2024-01-01T11:00:00Z"],
            "pit_verified": [True],
        }
    ).to_csv(tmp_path / "pit_replay_features.csv", index=False)

    history = pd.DataFrame(
        {
            "match_id": ["m2"],
            "home_goals": [2],
            "away_goals": [1],
        }
    )

    with pytest.raises((pd.errors.MergeError, RuntimeError)):
        _load_preflight_pit_features(tmp_path, history)
