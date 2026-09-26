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
            "feature_source_max_available_at_utc": ["2023-12-31T00:00:00Z", pd.NaT],
            "elo_diff": [10.0, -5.0],
        }
    ).to_csv(tmp_path / "pit_replay_features.csv", index=False)

    history = pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "home_goals": [2, 0],
            "away_goals": [1, 1],
            "source_available_at_utc": ["2023-12-31T00:00:00Z", "2024-01-01T00:00:00Z"],
        }
    )

    result = _load_preflight_pit_features(tmp_path, history)
    assert result is not None
    assert result["home_goals"].tolist() == [2, 0]
    assert result["away_goals"].tolist() == [1, 1]
    assert result["pit_verified"].tolist() == [True, False]
    assert result["target"].tolist() == [0, 2]


def test_load_preflight_pit_features_allows_unknown_publication_time_for_unverified_rows(tmp_path):
    pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "kickoff_utc": ["2024-01-01T12:00:00Z", "2024-01-02T12:00:00Z"],
            "prediction_cutoff_at_utc": ["2024-01-01T11:00:00Z", "2024-01-02T11:00:00Z"],
            "pit_verified": [True, False],
            "feature_source_max_available_at_utc": ["2023-12-31T00:00:00Z", pd.NaT],
        }
    ).to_csv(tmp_path / "pit_replay_features.csv", index=False)

    history = pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "home_goals": [1, 0],
            "away_goals": [0, 1],
            "source_available_at_utc": ["2023-12-31T00:00:00Z", pd.NaT],
        }
    )

    result = _load_preflight_pit_features(tmp_path, history)
    assert result is not None
    assert result["pit_verified"].tolist() == [True, False]
    assert pd.isna(result["source_available_at_utc"].iloc[1])


def test_load_preflight_pit_features_allows_missing_target_publication_time_for_verified_row(tmp_path):
    pd.DataFrame(
        {
            "match_id": ["m1"],
            "kickoff_utc": ["2024-01-01T12:00:00Z"],
            "prediction_cutoff_at_utc": ["2024-01-01T11:00:00Z"],
            "pit_verified": [True],
            "feature_source_max_available_at_utc": ["2023-12-31T00:00:00Z"],
        }
    ).to_csv(tmp_path / "pit_replay_features.csv", index=False)

    history = pd.DataFrame(
        {
            "match_id": ["m1"],
            "home_goals": [1],
            "away_goals": [0],
            "source_available_at_utc": [pd.NaT],
        }
    )

    result = _load_preflight_pit_features(tmp_path, history)
    assert result["pit_verified"].tolist() == [True]
    assert pd.isna(result["source_available_at_utc"].iloc[0])

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



def test_load_preflight_pit_features_restores_source_publication_time(tmp_path):
    pd.DataFrame(
        {
            "match_id": ["m1"],
            "kickoff_utc": ["2024-01-01T12:00:00Z"],
            "prediction_cutoff_at_utc": ["2024-01-01T11:00:00Z"],
            "pit_verified": [True],
            "feature_source_max_available_at_utc": ["2023-12-31T00:00:00Z"],
            "elo_diff": [10.0],
        }
    ).to_csv(tmp_path / "pit_replay_features.csv", index=False)

    history = pd.DataFrame(
        {
            "match_id": ["m1"],
            "home_goals": [2],
            "away_goals": [1],
            "source_available_at_utc": ["2023-12-31T00:00:00Z"],
        }
    )

    result = _load_preflight_pit_features(tmp_path, history)
    assert result["source_available_at_utc"].iloc[0] == pd.Timestamp("2023-12-31T00:00:00Z")


def test_load_preflight_pit_features_rejects_feature_availability_after_prediction_cutoff(tmp_path):
    pd.DataFrame(
        {
            "match_id": ["m1"],
            "kickoff_utc": ["2024-01-01T12:00:00Z"],
            "prediction_cutoff_at_utc": ["2024-01-01T11:00:00Z"],
            "pit_verified": [True],
            "feature_source_max_available_at_utc": ["2024-01-01T12:00:01Z"],
        }
    ).to_csv(tmp_path / "pit_replay_features.csv", index=False)

    history = pd.DataFrame(
        {
            "match_id": ["m1"],
            "home_goals": [1],
            "away_goals": [0],
            "source_available_at_utc": [pd.NaT],
        }
    )

    with pytest.raises(RuntimeError, match="feature_source_max_available_at_utc"):
        _load_preflight_pit_features(tmp_path, history)
