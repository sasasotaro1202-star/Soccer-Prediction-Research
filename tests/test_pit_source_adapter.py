import pandas as pd

from src.data.pit_source_adapter import (
    COMPETITION_ADAPTERS,
    FootballDataWaybackAdapter,
    SourceEvidence,
    competition_adapter_matrix,
    source_url,
)


def test_all_15_competitions_are_explicitly_classified():
    assert len(COMPETITION_ADAPTERS) == 15
    matrix = competition_adapter_matrix()
    assert len(matrix) == 15
    assert set(matrix["competition"]) == set(COMPETITION_ADAPTERS)


def test_source_url_uses_fixed_and_acquisition_mappings():
    assert source_url("E0", 2025).endswith("/2526/E0.csv")
    assert source_url("CH", 2025).endswith("/2526/E1.csv")
    assert source_url("EPL", 2025).endswith("/2526/E0.csv")
    assert source_url("CHA", 2025).endswith("/2526/E1.csv")


def test_unknown_competition_fails_closed():
    try:
        source_url("UCL", 2025)
    except ValueError as exc:
        assert "No PIT source mapping" in str(exc)
    else:
        raise AssertionError("unverified competition must not receive a source URL")


def test_row_key_contains_completed_result_identity():
    row = pd.Series({
        "home_team": "Team A",
        "away_team": "Team B",
        "kickoff_utc": "2025-09-01T18:00:00Z",
        "home_goals": 2,
        "away_goals": 1,
        "result": "H",
    })
    assert FootballDataWaybackAdapter._row_key(row) == ("2025-09-01", "Team A", "Team B", 2.0, 1.0, "H")


def test_source_evidence_never_invents_timestamp():
    evidence = SourceEvidence(None, "UNVERIFIABLE", reason="no_archive_snapshot_contains_completed_result")
    assert evidence.source_available_at_utc is None


def test_cdx_no_capture_is_distinct_from_request_failure(tmp_path, monkeypatch):
    adapter = FootballDataWaybackAdapter(cache_dir=str(tmp_path))
    monkeypatch.setattr("src.data.pit_source_adapter.requests.get", lambda *a, **k: (_ for _ in ()).throw(__import__("requests").RequestException("network down")))
    url = "https://example.invalid/test.csv"
    assert adapter.captures(url) == []
    assert adapter.capture_diagnostic(url).status == "CDX_REQUEST_FAILURE"


def test_cdx_empty_response_is_not_request_failure(tmp_path, monkeypatch):
    class Response:
        def raise_for_status(self):
            return None
        def json(self):
            return [["timestamp", "digest", "original", "statuscode", "mimetype"]]

    adapter = FootballDataWaybackAdapter(cache_dir=str(tmp_path))
    monkeypatch.setattr("src.data.pit_source_adapter.requests.get", lambda *a, **k: Response())
    url = "https://example.invalid/test.csv"
    assert adapter.captures(url) == []
    assert adapter.capture_diagnostic(url).status == "CDX_NO_CAPTURE"


def test_snapshot_parse_and_key_matching_are_observable(tmp_path, monkeypatch):
    csv = b"Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n01/09/25,Team A,Team B,2,1,H\n"

    class Response:
        content = csv
        def raise_for_status(self):
            return None

    adapter = FootballDataWaybackAdapter(cache_dir=str(tmp_path))
    monkeypatch.setattr("src.data.pit_source_adapter.requests.get", lambda *a, **k: Response())
    capture = {"timestamp": "20250902000000", "digest": "digest-a"}
    diag = adapter._load_snapshot_keys(capture, "https://example.invalid/test.csv")
    assert diag.status == "SNAPSHOT_PARSED"
    assert ("2025-09-01", "Team A", "Team B", 2.0, 1.0, "H") in diag.keys


def test_diagnostic_bulk_exposes_cdx_failure_stage(tmp_path, monkeypatch):
    adapter = FootballDataWaybackAdapter(cache_dir=str(tmp_path))
    monkeypatch.setattr("src.data.pit_source_adapter.requests.get", lambda *a, **k: (_ for _ in ()).throw(__import__("requests").RequestException("network down")))
    history = pd.DataFrame([{
        "competition": "EPL",
        "season": "2025/26",
        "kickoff_utc": "2025-09-01T18:00:00Z",
        "home_team": "Team A",
        "away_team": "Team B",
        "home_goals": 2,
        "away_goals": 1,
        "result": "H",
    }])
    diagnostic = adapter.diagnostic_bulk(history)
    assert diagnostic.loc[0, "cdx_status"] == "CDX_REQUEST_FAILURE"
    assert diagnostic.loc[0, "failure_stage"] == "CDX_REQUEST_FAILURE"
    assert "network down" in diagnostic.loc[0, "failure_reason"]
