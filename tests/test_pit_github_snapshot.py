from datetime import datetime, timezone

from src.data.pit_github_snapshot import (
    GitHubSnapshotEvidence,
    _snapshot_keys,
    dataset_path,
    evidence_for_row,
)


def test_dataset_paths_are_explicit_and_not_silent_fallbacks():
    assert dataset_path("EPL", 2010) == "2010-11/en.1.json"
    assert dataset_path("BL1", 2010) == "2010-11/de.1.json"
    assert dataset_path("SA", 2010) == "2010-11/it.1.json"
    assert dataset_path("LL", 2010) == "2010-11/es.1.json"
    assert dataset_path("FL1", 2010) == "2010-11/fr.1.json"


def test_snapshot_key_requires_completed_full_time_score():
    payload = {
        "matches": [
            {"date": "2015-08-08", "team1": "A", "team2": "B", "score": {"ft": [2, 1]}},
            {"date": "2015-08-09", "team1": "C", "team2": "D", "score": [0, 0]},
        ]
    }
    assert ("2015-08-08", "A", "B", 2.0, 1.0) in _snapshot_keys(payload)
    assert ("2015-08-09", "C", "D", 0.0, 0.0) not in _snapshot_keys(payload)


def test_missing_event_time_is_fail_closed():
    result = evidence_for_row("EPL", 2010, "", "A", "B", 1, 0)
    assert isinstance(result, GitHubSnapshotEvidence)
    assert result.status == "UNVERIFIABLE"
    assert result.reason == "missing_event_time"
