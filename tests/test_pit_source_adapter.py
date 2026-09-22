import pandas as pd

from src.data.pit_source_adapter import (
    COMPETITION_ADAPTERS,
    FootballDataWaybackAdapter,
    SourceEvidence,
    _result_lower_bound,
    competition_adapter_matrix,
    source_url,
)
from src.data.pit_archive_fallback import _captures


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


def test_row_key_uses_source_local_event_date_when_available():
    row = pd.Series({"home_team":"Aston Villa","away_team":"West Ham","source_event_date":"2010-08-14","kickoff_utc":"2010-08-13T23:00:00Z","home_goals":3,"away_goals":0,"result":"H"})
    assert FootballDataWaybackAdapter._row_key(row) == ("2010-08-14", "Aston Villa", "West Ham", 3.0, 0.0, "H")


def test_row_key_falls_back_to_kickoff_date_when_source_date_missing():
    row = pd.Series({"home_team":"Team A","away_team":"Team B","kickoff_utc":"2025-09-01T18:00:00Z","home_goals":2,"away_goals":1,"result":"H"})
    assert FootballDataWaybackAdapter._row_key(row) == ("2025-09-01", "Team A", "Team B", 2.0, 1.0, "H")


def test_source_evidence_never_invents_timestamp():
    evidence = SourceEvidence(None, "UNVERIFIABLE", reason="no_archive_snapshot_contains_completed_result")
    assert evidence.source_available_at_utc is None


def test_cdx_no_capture_is_distinct_from_request_failure(tmp_path, monkeypatch):
    adapter = FootballDataWaybackAdapter(cache_dir=str(tmp_path))
    monkeypatch.setattr("src.data.pit_source_adapter_fast.requests.get", lambda *a, **k: (_ for _ in ()).throw(__import__("requests").RequestException("network down")))
    assert adapter.captures("https://example.invalid/test.csv") == []
    assert adapter.capture_diagnostic("https://example.invalid/test.csv").status == "CDX_REQUEST_FAILURE"


def test_cdx_empty_response_is_not_request_failure(tmp_path, monkeypatch):
    class Response:
        def raise_for_status(self): return None
        def json(self): return [["timestamp", "digest", "original", "statuscode", "mimetype"]]
    adapter = FootballDataWaybackAdapter(cache_dir=str(tmp_path))
    monkeypatch.setattr("src.data.pit_source_adapter_fast.requests.get", lambda *a, **k: Response())
    assert adapter.captures("https://example.invalid/test.csv") == []
    assert adapter.capture_diagnostic("https://example.invalid/test.csv").status == "CDX_NO_CAPTURE"


def test_snapshot_parse_and_key_matching_are_observable(tmp_path, monkeypatch):
    csv = b"Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n01/09/25,Team A,Team B,2,1,H\n"
    class Response:
        content = csv
        def raise_for_status(self): return None
    adapter = FootballDataWaybackAdapter(cache_dir=str(tmp_path))
    monkeypatch.setattr("src.data.pit_source_adapter_fast.requests.get", lambda *a, **k: Response())
    capture = {"timestamp":"20250902000000","digest":"digest-a"}
    diag = adapter._load_snapshot_keys(capture, "https://example.invalid/test.csv")
    assert diag.status == "SNAPSHOT_PARSED"
    assert ("2025-09-01", "Team A", "Team B", 2.0, 1.0, "H") in diag.keys


def test_diagnostic_bulk_exposes_cdx_failure_stage(tmp_path, monkeypatch):
    adapter = FootballDataWaybackAdapter(cache_dir=str(tmp_path))
    monkeypatch.setattr("src.data.pit_source_adapter_fast.requests.get", lambda *a, **k: (_ for _ in ()).throw(__import__("requests").RequestException("network down")))
    history = pd.DataFrame([{"competition":"EPL","season":"2025/26","kickoff_utc":"2025-09-01T18:00:00Z","kickoff_time_available":True,"home_team":"Team A","away_team":"Team B","home_goals":2,"away_goals":1,"result":"H"}])
    diagnostic = adapter.diagnostic_bulk(history)
    assert diagnostic.loc[0, "cdx_status"] == "CDX_REQUEST_FAILURE"
    assert diagnostic.loc[0, "failure_stage"] == "CDX_REQUEST_FAILURE"
    assert "network down" in diagnostic.loc[0, "failure_reason"]


def test_precise_kickoff_uses_conservative_completion_lower_bound():
    row = pd.Series({"kickoff_utc":"2025-09-01T18:00:00Z","kickoff_time_available":True})
    bound, reason = _result_lower_bound(row)
    assert bound.isoformat() == "2025-09-01T21:00:00+00:00"
    assert reason == "KICKOFF_PLUS_180M"


def test_date_only_does_not_claim_same_day_result_availability():
    row = pd.Series({"kickoff_utc":"2025-09-01T00:00:00Z","kickoff_time_available":False})
    bound, reason = _result_lower_bound(row)
    assert bound.isoformat() == "2025-09-02T00:00:00+00:00"
    assert reason == "DATE_ONLY_NEXT_DAY"


def test_replay_rejects_completed_result_observed_before_180m_even_after_kickoff(tmp_path, monkeypatch):
    adapter = FootballDataWaybackAdapter(cache_dir=str(tmp_path))
    monkeypatch.setattr(
        adapter,
        "captures",
        lambda url: [{"timestamp":"20250901193000","digest":"digest-early","original":url}],
    )
    # Isolate the replay-window acceptance rule from HTTP/CSV parsing. Snapshot
    # parsing is covered independently above; this test should fail only when
    # the PIT cutoff logic regresses.
    monkeypatch.setattr(
        adapter,
        "_load_snapshot_keys",
        lambda capture, original_url: type("Diag", (), {
            "status": "SNAPSHOT_PARSED",
            "keys": {("2025-09-01", "teama", "teamb", 2.0, 1.0, "H")},
        })(),
    )
    row = pd.Series({"competition":"EPL","season_start":2025,"home_team":"Team A","away_team":"Team B","source_event_date":"2025-09-01","kickoff_utc":"2025-09-01T18:00:00Z","kickoff_time_available":True,"home_goals":2,"away_goals":1,"result":"H"})
    evidence = adapter._prefetch_url("https://example.invalid/test.csv", [row], workers=1)[0]
    assert evidence.evidence_status == "UNVERIFIABLE"
    assert evidence.source_available_at_utc is None


def test_arquivo_cdx_mapping_error_is_treated_as_no_capture(monkeypatch):
    class Response:
        def raise_for_status(self): return None
        def json(self): return {"message":"temporarily unavailable"}
    monkeypatch.setattr("src.data.pit_archive_fallback.requests.get", lambda *a, **k: Response())
    assert _captures("https://example.invalid/test.csv", retries=1, timeout=1) == []


def test_snapshot_retries_exact_capture_original_url_fallback(tmp_path, monkeypatch):
    csv = b"Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n01/09/25,Team A,Team B,2,1,H\n"
    calls = []

    class Response:
        content = csv
        def raise_for_status(self):
            return None

    def fake_get(url, *args, **kwargs):
        calls.append(url)
        if len(calls) == 1:
            raise __import__("requests").RequestException("primary replay unavailable")
        return Response()

    adapter = FootballDataWaybackAdapter(cache_dir=str(tmp_path), snapshot_retries=1)
    monkeypatch.setattr("src.data.pit_source_adapter.requests.get", fake_get)
    capture = {
        "timestamp": "20250902200000",
        "digest": "digest-a",
        "original": "https://example.invalid/canonical.csv",
    }
    diag = adapter._load_snapshot_keys(capture, "https://example.invalid/test.csv")
    assert diag.status == "SNAPSHOT_PARSED"
    assert len(calls) == 2
    assert calls[0] != calls[1]
    assert any("canonical.csv" in url for url in calls)
