from __future__ import annotations

import json

import pandas as pd
import pytest

from src.prediction.runner import _require_fresh_matchday_snapshot


def _write_status(tmp_path, payload):
    path = tmp_path / "matchday_status.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_fresh_matchday_accepts_clean_collected_status(tmp_path):
    path = _write_status(
        tmp_path,
        {
            "status": "COLLECTED",
            "errors": [],
            "rows": 2,
            "snapshot_finished_at_utc": "2026-10-04T08:00:00Z",
        },
    )
    result = _require_fresh_matchday_snapshot(
        path,
        pd.Timestamp("2026-10-04T08:05:00Z"),
        max_age_minutes=15.0,
    )
    assert result["status"] == "FRESH"


def test_fresh_matchday_rejects_collected_status_with_errors(tmp_path):
    path = _write_status(
        tmp_path,
        {
            "status": "COLLECTED",
            "errors": [{"source": "espn_scoreboard", "error": "unavailable"}],
            "rows": 4,
            "snapshot_finished_at_utc": "2026-10-04T08:00:00Z",
        },
    )
    with pytest.raises(RuntimeError, match="internally inconsistent"):
        _require_fresh_matchday_snapshot(
            path,
            pd.Timestamp("2026-10-04T08:05:00Z"),
            max_age_minutes=15.0,
        )


def test_fresh_matchday_rejects_partial_status_without_successful_discovery_source(tmp_path):
    path = _write_status(
        tmp_path,
        {
            "status": "COLLECTED_WITH_ERRORS",
            "errors": [{"source": "espn_scoreboard", "error": "unavailable"}],
            "discovery_sources_observed": [],
            "discovery_source_redundancy_ok": False,
            "rows": 3,
            "snapshot_finished_at_utc": "2026-10-04T08:00:00Z",
        },
    )
    with pytest.raises(RuntimeError, match="successful discovery source"):
        _require_fresh_matchday_snapshot(
            path,
            pd.Timestamp("2026-10-04T08:05:00Z"),
            max_age_minutes=15.0,
        )


def test_fresh_matchday_accepts_partial_status_with_successful_discovery_source(tmp_path):
    path = _write_status(
        tmp_path,
        {
            "status": "COLLECTED_WITH_ERRORS",
            "errors": [{"source": "espn_scoreboard", "error": "unavailable"}],
            "discovery_sources_observed": ["sofascore"],
            "discovery_source_redundancy_ok": True,
            "rows": 6,
            "snapshot_finished_at_utc": "2026-10-04T08:00:00Z",
        },
    )
    result = _require_fresh_matchday_snapshot(
        path,
        pd.Timestamp("2026-10-04T08:05:00Z"),
        max_age_minutes=15.0,
    )
    assert result["status"] == "FRESH"


def test_matchday_collector_exposes_explicit_partial_source_state():
    from pathlib import Path

    source = Path("src/data/matchday_intelligence_fetch.py").read_text(encoding="utf-8")
    assert '"COLLECTED_WITH_ERRORS"' in source
    assert '"discovery_sources_observed"' in source
    assert '"discovery_source_errors"' in source
