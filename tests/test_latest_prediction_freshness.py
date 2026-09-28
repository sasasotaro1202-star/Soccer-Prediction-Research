import json

import pandas as pd
import pytest

from src.prediction.runner import _require_fresh_matchday_snapshot


def test_latest_prediction_accepts_fresh_collected_snapshot(tmp_path):
    status = tmp_path / "matchday_status.json"
    status.write_text(
        json.dumps({
            "status": "COLLECTED",
            "snapshot_finished_at_utc": "2026-09-29T06:30:00Z",
        }),
        encoding="utf-8",
    )
    result = _require_fresh_matchday_snapshot(
        str(status),
        pd.Timestamp("2026-09-29T06:38:00Z"),
        max_age_minutes=15.0,
    )
    assert result["status"] == "FRESH"
    assert result["age_minutes"] == 8.0


def test_latest_prediction_rejects_stale_snapshot(tmp_path):
    status = tmp_path / "matchday_status.json"
    status.write_text(
        json.dumps({
            "status": "COLLECTED",
            "snapshot_finished_at_utc": "2026-09-29T06:00:00Z",
        }),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="stale"):
        _require_fresh_matchday_snapshot(
            str(status),
            pd.Timestamp("2026-09-29T06:20:01Z"),
            max_age_minutes=15.0,
        )


def test_latest_prediction_rejects_non_collected_snapshot(tmp_path):
    status = tmp_path / "matchday_status.json"
    status.write_text(
        json.dumps({
            "status": "DEFERRED_EXTERNAL_SOURCE",
            "snapshot_finished_at_utc": "2026-09-29T06:35:00Z",
        }),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="successful current matchday refresh"):
        _require_fresh_matchday_snapshot(
            str(status),
            pd.Timestamp("2026-09-29T06:38:00Z"),
            max_age_minutes=15.0,
        )


def test_latest_prediction_rejects_missing_status(tmp_path):
    with pytest.raises(RuntimeError, match="refusing stale artifact reuse"):
        _require_fresh_matchday_snapshot(
            str(tmp_path / "missing.json"),
            pd.Timestamp("2026-09-29T06:38:00Z"),
        )
