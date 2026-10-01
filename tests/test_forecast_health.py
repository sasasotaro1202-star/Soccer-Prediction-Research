from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.prediction.research_forecast import FORECAST_COLUMNS
from src.research.forecast_health import audit


def _write_status(path: Path, **overrides):
    value = {
        "status": "PREDICTED_PRODUCTION_ADOPTED",
        "prediction_time_utc": "2026-10-01T08:00:00Z",
        "rows": 2,
        "prediction_rows": 2,
        "production_model_used": True,
        "research_heuristic_disabled": True,
        "runner_status": {
            "status": "PREDICTED",
            "freshness": {
                "status": "FRESH",
                "age_minutes": 4.0,
                "max_age_minutes": 15.0,
            },
        },
    }
    value.update(overrides)
    path.write_text(json.dumps(value), encoding="utf-8")


def _forecast_row(match_id: str, kickoff: str):
    return {
        "match_id": match_id,
        "kickoff_utc": kickoff,
        "competition": "EPL",
        "home_team": "Alpha",
        "away_team": "Beta",
        "home_win_probability": 0.55,
        "draw_probability": 0.25,
        "away_win_probability": 0.20,
        "result_prediction": "Alpha",
        "result_prediction_probability": 0.55,
        "score_1": "1-0",
        "score_1_probability": 0.20,
        "score_2": "1-1",
        "score_2_probability": 0.15,
        "score_3": "2-0",
        "score_3_probability": 0.10,
        "over_2_5_probability": 0.48,
        "under_2_5_probability": 0.52,
        "btts_yes_probability": 0.51,
        "btts_no_probability": 0.49,
        "mom_status": "BLOCKED_UPSTREAM_PLAYER_MODEL",
        "mom_method": "test",
        "mom_1_player": "",
        "mom_1_probability": float("nan"),
        "mom_2_player": "",
        "mom_2_probability": float("nan"),
        "mom_3_player": "",
        "mom_3_probability": float("nan"),
        "mom_4_player": "",
        "mom_4_probability": float("nan"),
    }


def _write_forecast(path: Path, rows):
    pd.DataFrame(rows, columns=FORECAST_COLUMNS).to_csv(path, index=False)


def test_healthy_prediction_requires_nonempty_and_future_rows(tmp_path):
    output = tmp_path / "forecast.csv"
    status = tmp_path / "status.json"
    _write_status(status)
    _write_forecast(
        output,
        [
            _forecast_row("m1", "2026-10-01T09:00:00Z"),
            _forecast_row("m2", "2026-10-01T10:00:00Z"),
        ],
    )

    result = audit(str(output), str(status))
    assert result["status"] == "HEALTHY"
    assert result["ok"] is True
    assert result["prediction_runtime_ready"] is True
    assert result["coverage_ratio"] == 1.0


def test_empty_output_cannot_be_marked_predicted(tmp_path):
    output = tmp_path / "forecast.csv"
    status = tmp_path / "status.json"
    _write_status(status)
    _write_forecast(output, [])

    result = audit(str(output), str(status))
    assert result["ok"] is False
    assert "predicted_status_requires_nonempty_forecast" in result["failures"]


def test_no_target_is_distinguished_from_verified_prediction(tmp_path):
    output = tmp_path / "forecast.csv"
    status = tmp_path / "status.json"
    _write_status(
        status,
        status="NO_TARGET_FIXTURES",
        rows=0,
        prediction_rows=0,
        production_model_used=False,
    )
    _write_forecast(output, [])

    result = audit(str(output), str(status))
    assert result["status"] == "NO_TARGET"
    assert result["ok"] is True
    assert result["prediction_runtime_ready"] is False
    assert result["forecast_evidence"] == "NONE"


def test_deferred_model_is_blocked(tmp_path):
    output = tmp_path / "forecast.csv"
    status = tmp_path / "status.json"
    _write_status(
        status,
        status="DEFERRED_NO_ADOPTED_MODEL",
        rows=3,
        prediction_rows=0,
        production_model_used=False,
    )
    _write_forecast(output, [])

    result = audit(str(output), str(status))
    assert result["status"] == "BLOCKED"
    assert result["ok"] is False


def test_pit_violation_blocks_prediction(tmp_path):
    output = tmp_path / "forecast.csv"
    status = tmp_path / "status.json"
    _write_status(status)
    _write_forecast(output, [_forecast_row("m1", "2026-10-01T07:59:00Z")])

    result = audit(str(output), str(status))
    assert result["ok"] is False
    assert "prediction_contains_non_future_kickoff" in result["failures"]


def test_probability_corruption_blocks_prediction(tmp_path):
    output = tmp_path / "forecast.csv"
    status = tmp_path / "status.json"
    _write_status(status)
    row = _forecast_row("m1", "2026-10-01T09:00:00Z")
    row["draw_probability"] = 0.60
    _write_forecast(output, [row])

    result = audit(str(output), str(status))
    assert result["ok"] is False
    assert any("1X2 probabilities" in x for x in result["failures"])




def test_missing_freshness_evidence_blocks_predicted_status(tmp_path):
    output = tmp_path / "forecast.csv"
    status = tmp_path / "status.json"
    _write_status(status)
    payload = json.loads(status.read_text(encoding="utf-8"))
    payload["runner_status"]["freshness"]["status"] = "STALE"
    status.write_text(json.dumps(payload), encoding="utf-8")
    _write_forecast(output, [_forecast_row("m1", "2026-10-01T09:00:00Z")])

    result = audit(str(output), str(status))
    assert result["ok"] is False
    assert "prediction_requires_FRESH_matchday_snapshot" in result["failures"]


def test_missing_runner_status_blocks_predicted_status(tmp_path):
    output = tmp_path / "forecast.csv"
    status = tmp_path / "status.json"
    _write_status(status)
    payload = json.loads(status.read_text(encoding="utf-8"))
    payload.pop("runner_status")
    status.write_text(json.dumps(payload), encoding="utf-8")
    _write_forecast(output, [_forecast_row("m1", "2026-10-01T09:00:00Z")])

    result = audit(str(output), str(status))
    assert result["ok"] is False
    assert "predicted_status_requires_runner_status_evidence" in result["failures"]


def test_missing_status_is_fail_closed(tmp_path):
    output = tmp_path / "forecast.csv"
    _write_forecast(output, [])
    result = audit(str(output), str(tmp_path / "missing.json"))
    assert result["ok"] is False
    assert result["status"] == "BLOCKED"
