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


def test_commit_history_paginates_beyond_first_100(monkeypatch):
    import src.data.pit_github_snapshot as module

    calls = []

    class Response:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fake_request(url, timeout=None):
        calls.append(url)
        page = int(url.rsplit("page=", 1)[1])
        if page == 1:
            return Response([{"sha": f"sha-{i}"} for i in range(100)])
        if page == 2:
            return Response([{"sha": "sha-100"}])
        raise AssertionError(f"unexpected page: {page}")

    monkeypatch.setattr(module, "_request", fake_request)

    commits = module._commits("2010-11/en.1.json")

    assert len(commits) == 101
    assert commits[-1]["sha"] == "sha-100"
    assert calls[0].endswith("page=1")
    assert calls[1].endswith("page=2")


def test_commit_history_api_malformed_page_is_fail_closed(monkeypatch):
    import src.data.pit_github_snapshot as module

    class Response:
        def json(self):
            return {"unexpected": "shape"}

    monkeypatch.setattr(module, "_request", lambda url, timeout=None: Response())

    import pytest
    with pytest.raises(ValueError, match="unexpected GitHub commits response on page 1"):
        module._commits("2010-11/en.1.json")
