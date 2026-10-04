from __future__ import annotations

import pandas as pd

from src.data.pit_source_adapter_fast import (
    FootballDataWaybackAdapter,
    SnapshotDiagnostic,
    normalize_team_identity,
)


def _row():
    return pd.Series(
        {
            "competition": "EPL",
            "season_start": 2024,
            "source_event_date": "2024-01-10",
            "kickoff_utc": "2024-01-10T10:00:00Z",
            "kickoff_time_available": True,
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "home_goals": 2,
            "away_goals": 1,
            "result": "H",
        }
    )


def _key(row):
    return (
        "2024-01-10",
        normalize_team_identity(row["home_team"]),
        normalize_team_identity(row["away_team"]),
        2.0,
        1.0,
        "H",
    )


def test_exact_kickoff_capture_is_verified_without_relaxing_before_kickoff():
    adapter = FootballDataWaybackAdapter(cache_dir="/tmp/soccer-pit-fast-test", max_workers=1)
    row = _row()
    capture = {
        "timestamp": "20240110103000",
        "digest": "digest-after-kickoff",
        "original": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    }
    adapter.captures = lambda url: [capture]
    adapter._load_snapshot_keys = lambda cap, url: SnapshotDiagnostic(
        "SNAPSHOT_PARSED",
        keys={_key(row)},
    )

    evidence = adapter._prefetch_url("https://example.test/E0.csv", [row], workers=1)[0]

    assert evidence.evidence_status == "VERIFIED"
    assert evidence.source_available_at_utc == "2024-01-10T10:30:00+00:00"
    assert "after_kickoff" in evidence.reason


def test_exact_kickoff_capture_before_kickoff_is_not_accepted():
    adapter = FootballDataWaybackAdapter(cache_dir="/tmp/soccer-pit-fast-test", max_workers=1)
    row = _row()
    capture = {
        "timestamp": "20240110093000",
        "digest": "digest-before-kickoff",
        "original": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    }
    adapter.captures = lambda url: [capture]
    adapter._load_snapshot_keys = lambda cap, url: SnapshotDiagnostic(
        "SNAPSHOT_PARSED",
        keys={_key(row)},
    )

    evidence = adapter._prefetch_url("https://example.test/E0.csv", [row], workers=1)[0]

    assert evidence.evidence_status == "UNVERIFIABLE"
    assert evidence.source_available_at_utc is None
