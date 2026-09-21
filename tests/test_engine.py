import json

import pandas as pd

from src.research.engine import _load_authoritative_pit_features


def _history():
    return pd.DataFrame([
        {
            "match_id": "m1",
            "competition": "EPL",
            "season_start": 2024,
            "kickoff_utc": pd.Timestamp("2025-01-01T18:00:00Z"),
            "home_team": "A",
            "away_team": "B",
        },
        {
            "match_id": "m2",
            "competition": "EPL",
            "season_start": 2024,
            "kickoff_utc": pd.Timestamp("2025-01-02T18:00:00Z"),
            "home_team": "B",
            "away_team": "A",
        },
    ])


def test_authoritative_pit_features_are_reused_when_gate_matches(tmp_path):
    history = _history()
    features = history.copy()
    features["pit_verified"] = [True, False]
    features["feature_value"] = [1.0, 2.0]
    features.to_csv(tmp_path / "pit_replay_features.csv", index=False)
    (tmp_path / "completion_gate.json").write_text(json.dumps({
        "full_gate_passed": True,
        "pit_publication_time_gate": True,
        "pit_replay_rows": 2,
        "pit_replay_verified_rows": 1,
    }), encoding="utf-8")

    loaded = _load_authoritative_pit_features(history, tmp_path)

    assert loaded is not None
    assert loaded["feature_value"].tolist() == [1.0, 2.0]
    assert loaded["pit_verified"].tolist() == [True, False]


def test_authoritative_pit_features_reject_stale_identity(tmp_path):
    history = _history()
    features = history.copy()
    features.loc[0, "home_team"] = "STALE"
    features["pit_verified"] = [True, True]
    features.to_csv(tmp_path / "pit_replay_features.csv", index=False)
    (tmp_path / "completion_gate.json").write_text(json.dumps({
        "full_gate_passed": True,
        "pit_publication_time_gate": True,
        "pit_replay_rows": 2,
        "pit_replay_verified_rows": 2,
    }), encoding="utf-8")

    assert _load_authoritative_pit_features(history, tmp_path) is None


def test_authoritative_pit_features_reject_gate_mismatch(tmp_path):
    history = _history()
    features = history.copy()
    features["pit_verified"] = [True, True]
    features.to_csv(tmp_path / "pit_replay_features.csv", index=False)
    (tmp_path / "completion_gate.json").write_text(json.dumps({
        "full_gate_passed": False,
        "pit_publication_time_gate": False,
        "pit_replay_rows": 2,
        "pit_replay_verified_rows": 2,
    }), encoding="utf-8")

    assert _load_authoritative_pit_features(history, tmp_path) is None
