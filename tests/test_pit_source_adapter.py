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
    monkeypatch.setattr("src.data.pit_source_adapter.resilient_get", lambda getter, url, **kwargs: fake_get(url, **kwargs))
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


def test_replay_capture_fetch_is_bounded_and_stops_after_all_rows_resolve(tmp_path, monkeypatch):
    adapter = FootballDataWaybackAdapter(cache_dir=str(tmp_path))
    captures = [
        {"timestamp": "20250901213000", "digest": "digest-1", "original": "https://example.invalid/test.csv"},
        {"timestamp": "20250901220000", "digest": "digest-2", "original": "https://example.invalid/test.csv"},
        {"timestamp": "20250901223000", "digest": "digest-3", "original": "https://example.invalid/test.csv"},
    ]
    calls = []

    monkeypatch.setenv("PIT_CAPTURE_BATCH_SIZE", "1")
    monkeypatch.setattr(adapter, "captures", lambda url: captures)

    def fake_snapshot(capture, original_url):
        calls.append(capture["digest"])
        keys = (
            {("2025-09-01", "teama", "teamb", 2.0, 1.0, "H")}
            if capture["digest"] == "digest-2"
            else set()
        )
        return type("Diag", (), {"status": "SNAPSHOT_PARSED", "keys": keys})()

    monkeypatch.setattr(adapter, "_load_snapshot_keys", fake_snapshot)
    row = pd.Series({
        "competition": "EPL",
        "season_start": 2025,
        "source_event_date": "2025-09-01",
        "kickoff_utc": "2025-09-01T18:00:00Z",
        "kickoff_time_available": True,
        "home_team": "Team A",
        "away_team": "Team B",
        "home_goals": 2,
        "away_goals": 1,
        "result": "H",
    })

    evidence = adapter._prefetch_url("https://example.invalid/test.csv", [row], workers=1)[0]

    assert evidence.evidence_status == "VERIFIED"
    assert evidence.source_available_at_utc == "2025-09-01T22:00:00+00:00"
    assert calls == ["digest-1", "digest-2"]


def test_malformed_snapshot_cache_is_invalidated_and_refetched(tmp_path, monkeypatch):
    csv = b"Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n01/09/25,Team A,Team B,2,1,H\n"
    calls = []

    class Response:
        content = csv
        def raise_for_status(self):
            return None

    adapter = FootballDataWaybackAdapter(
        cache_dir=str(tmp_path),
        snapshot_retries=2,
        retry_backoff=0,
    )
    capture = {
        "timestamp": "20250902200000",
        "digest": "digest-corrupt",
        "original": "https://example.invalid/test.csv",
    }
    cache = adapter._snapshot_cache_path(capture)
    cache.write_bytes(b"<html>temporary gateway error</html>")

    def fake_get(url, *args, **kwargs):
        calls.append(url)
        return Response()

    monkeypatch.setattr("src.data.pit_source_adapter.resilient_get", lambda getter, url, **kwargs: fake_get(url, **kwargs))

    diag = adapter._load_snapshot_keys(capture, "https://example.invalid/test.csv")

    assert diag.status == "SNAPSHOT_PARSED"
    assert len(calls) == 1
    assert cache.read_bytes() == csv

 
 
def test_compatibility_adapter_has_no_direct_requests_get_or_legacy_timeout():
    import inspect

    source = inspect.getsource(FootballDataWaybackAdapter)
    assert "requests.get(" not in source
    assert "max(self.timeout" not in source
    assert "resilient_get(" in source


def test_compatibility_adapter_default_timeout_none_reaches_resilience_layer(tmp_path, monkeypatch):
    csv = b"Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n01/09/25,Team A,Team B,2,1,H\n"
    observed = []

    class Response:
        content = csv

        def raise_for_status(self):
            return None

    def fake_resilient_get(getter, url, **kwargs):
        observed.append({"url": url, "timeout": kwargs["timeout"], "retries": kwargs["retries"]})
        return Response()

    monkeypatch.setattr("src.data.pit_source_adapter.resilient_get", fake_resilient_get)
    instance = FootballDataWaybackAdapter(
        cache_dir=str(tmp_path),
        timeout=None,
        snapshot_retries=1,
        snapshot_retry_backoff=0,
    )
    capture = {
        "timestamp": "20250902200000",
        "digest": "digest-none-timeout",
        "original": "https://example.invalid/test.csv",
    }

    diagnostic = instance._load_snapshot_keys(
        capture,
        "https://example.invalid/test.csv",
    )

    assert diagnostic.status == "SNAPSHOT_PARSED"
    assert observed
    assert observed[0]["timeout"] is None
    assert observed[0]["retries"] == 1


def test_verified_evidence_cache_resumes_without_new_archive_fetch(tmp_path, monkeypatch):
    adapter = FootballDataWaybackAdapter(cache_dir=str(tmp_path))
    row = pd.Series({
        "competition": "EPL",
        "season_start": 2025,
        "source_event_date": "2025-09-01",
        "kickoff_utc": "2025-09-01T18:00:00Z",
        "kickoff_time_available": True,
        "home_team": "Team A",
        "away_team": "Team B",
        "home_goals": 2,
        "away_goals": 1,
        "result": "H",
    })
    evidence = SourceEvidence(
        "2025-09-01T22:00:00+00:00",
        "VERIFIED",
        "https://web.archive.org/web/20250901220000id_/https://example.invalid/test.csv",
        "digest-2",
        "verified-test",
    )
    lower_bound, _ = _result_lower_bound(row)
    adapter._save_verified_evidence_cache("https://example.invalid/test.csv", row, lower_bound, evidence)

    def fail_fetch(*args, **kwargs):
        raise AssertionError("archive snapshot must not be fetched when VERIFIED cache is valid")

    monkeypatch.setattr(adapter, "captures", fail_fetch)
    resumed = adapter._prefetch_url("https://example.invalid/test.csv", [row], workers=1)[0]

    assert resumed.evidence_status == "VERIFIED"
    assert resumed.source_available_at_utc == evidence.source_available_at_utc
    assert resumed.evidence_url == evidence.evidence_url
    assert resumed.capture_digest == evidence.capture_digest


def test_unverifiable_evidence_is_not_cached_for_future_recovery(tmp_path):
    adapter = FootballDataWaybackAdapter(cache_dir=str(tmp_path))
    row = pd.Series({
        "competition": "EPL",
        "season_start": 2025,
        "source_event_date": "2025-09-01",
        "kickoff_utc": "2025-09-01T18:00:00Z",
        "kickoff_time_available": True,
        "home_team": "Team A",
        "away_team": "Team B",
        "home_goals": 2,
        "away_goals": 1,
        "result": "H",
    })
    lower_bound, _ = _result_lower_bound(row)
    evidence = SourceEvidence(None, "UNVERIFIABLE", reason="temporary archive outage")
    adapter._save_verified_evidence_cache("https://example.invalid/test.csv", row, lower_bound, evidence)
    assert not adapter._evidence_cache_path("https://example.invalid/test.csv", row, lower_bound).exists()


def test_verified_evidence_cache_revalidates_lower_bound(tmp_path):
    adapter = FootballDataWaybackAdapter(cache_dir=str(tmp_path))
    row = pd.Series({
        "competition": "EPL",
        "season_start": 2025,
        "source_event_date": "2025-09-01",
        "kickoff_utc": "2025-09-01T18:00:00Z",
        "kickoff_time_available": True,
        "home_team": "Team A",
        "away_team": "Team B",
        "home_goals": 2,
        "away_goals": 1,
        "result": "H",
    })
    bound = pd.Timestamp("2025-09-01T21:00:00Z")
    path = adapter._evidence_cache_path("https://example.invalid/test.csv", row, bound)
    path.write_text("{\"cache_version\":1,\"status\":\"VERIFIED\",\"url\":\"https://example.invalid/test.csv\",\"row_key\":[\"2025-09-01\",\"teama\",\"teamb\",2.0,1.0,\"H\"],\"source_available_at_utc\":\"2025-09-01T20:00:00+00:00\"}", encoding="utf-8")
    assert adapter._load_verified_evidence_cache("https://example.invalid/test.csv", row, bound) is None


def test_resumable_pit_replay_preserves_duplicate_row_keys(tmp_path, monkeypatch):
    adapter = FootballDataWaybackAdapter(cache_dir=str(tmp_path))
    captures = [{"timestamp": "20250901220000", "digest": "digest-1", "original": "https://example.invalid/test.csv"}]
    monkeypatch.setattr(adapter, "captures", lambda url: captures)
    monkeypatch.setenv("PIT_CAPTURE_BATCH_SIZE", "1")

    def fake_snapshot(capture, original_url):
        return type("Diag", (), {
            "status": "SNAPSHOT_PARSED",
            "keys": {("2025-09-01", "teama", "teamb", 2.0, 1.0, "H")},
        })()

    monkeypatch.setattr(adapter, "_load_snapshot_keys", fake_snapshot)
    rows = [pd.Series({
        "match_id": "source-a", "source_name": "same-source", "source_record_id": "1",
        "source_event_date": "2025-09-01", "kickoff_utc": "2025-09-01T18:00:00Z",
        "kickoff_time_available": True, "home_team": "Team A", "away_team": "Team B",
        "home_goals": 2, "away_goals": 1, "result": "H",
    }), pd.Series({
        "match_id": "source-b", "source_name": "same-source", "source_record_id": "2",
        "source_event_date": "2025-09-01", "kickoff_utc": "2025-09-01T18:00:00Z",
        "kickoff_time_available": True, "home_team": "Team A", "away_team": "Team B",
        "home_goals": 2, "away_goals": 1, "result": "H",
    })]
    result = adapter._prefetch_url("https://example.invalid/test.csv", rows, workers=1)
    assert [item.evidence_status for item in result] == ["VERIFIED", "VERIFIED"]
    assert [item.capture_digest for item in result] == ["digest-1", "digest-1"]
