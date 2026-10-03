from __future__ import annotations

import pandas as pd

from src.data.matchday_rich_enrichment import (
    _market_summary,
    _parse_weather_payload,
    _summarize_h2h,
    _summarize_recent_events,
)


def test_recent_event_summary_is_not_future_looking():
    kickoff = pd.Timestamp("2026-10-04T12:00:00Z")
    events = [
        {
            "startTimestamp": int(pd.Timestamp("2026-10-03T12:00:00Z").timestamp()),
            "status": {"type": "finished"},
            "homeTeam": {"id": 1},
            "awayTeam": {"id": 2},
            "homeScore": {"current": 2},
            "awayScore": {"current": 0},
        },
        {
            "startTimestamp": int(pd.Timestamp("2026-10-04T13:00:00Z").timestamp()),
            "status": {"type": "finished"},
            "homeTeam": {"id": 1},
            "awayTeam": {"id": 3},
            "homeScore": {"current": 4},
            "awayScore": {"current": 0},
        },
    ]
    out = _summarize_recent_events(events, "1", kickoff)
    assert out["matches"] == 1
    assert out["wins"] == 1
    assert out["gf"] == 2
    assert out["ga"] == 0
    assert out["matches_7d"] == 1


def test_recent_event_summary_missing_history_stays_unknown():
    out = _summarize_recent_events([], "1", pd.Timestamp("2026-10-04T12:00:00Z"))
    assert pd.isna(out["matches"])
    assert pd.isna(out["wins"])
    assert pd.isna(out["points"])


def test_market_summary_preserves_provider_dispersion():
    payload = {
        "odds": [
            {
                "provider": {"name": "A"},
                "homeTeamOdds": {"decimalValue": 2.0},
                "drawOdds": {"decimalValue": 3.5},
                "awayTeamOdds": {"decimalValue": 4.0},
            },
            {
                "provider": {"name": "B"},
                "homeTeamOdds": {"decimalValue": 2.5},
                "drawOdds": {"decimalValue": 3.2},
                "awayTeamOdds": {"decimalValue": 3.4},
            },
        ]
    }
    out = _market_summary([payload])
    assert out["rich_market_provider_count"] == 2
    assert out["rich_market_home_prob_std"] > 0
    assert set(out["rich_market_provider_names"].split(" | ")) == {"A", "B"}


def test_market_missing_stays_unknown():
    out = _market_summary([{"odds": []}])
    assert pd.isna(out["rich_market_provider_count"])
    assert out["rich_market_provider_names"] == ""


def test_weather_parser_keeps_rich_variables():
    kickoff = pd.Timestamp("2026-10-04T12:00:00Z")
    payload = {
        "hourly": {
            "time": ["2026-10-04T12:00:00Z"],
            "temperature_2m": [18],
            "relative_humidity_2m": [72],
            "dew_point_2m": [13],
            "apparent_temperature": [17],
            "precipitation_probability": [60],
            "precipitation": [1.2],
            "rain": [1.0],
            "showers": [0.0],
            "snowfall": [0.0],
            "visibility": [10000],
            "pressure_msl": [1012],
            "surface_pressure": [1006],
            "cloud_cover": [80],
            "cloud_cover_low": [60],
            "cloud_cover_mid": [50],
            "cloud_cover_high": [40],
            "wind_speed_10m": [22],
            "wind_direction_10m": [180],
            "wind_gusts_10m": [35],
            "is_day": [1],
            "sunshine_duration": [1800],
            "cape": [120],
            "weather_code": [61],
        }
    }
    out = _parse_weather_payload(payload, kickoff)
    assert out["rich_weather_relative_humidity_2m"] == 72.0
    assert out["rich_weather_wind_gusts_10m"] == 35.0
    assert out["rich_weather_weather_code"] == 61


def test_weather_missing_values_stay_unknown():
    kickoff = pd.Timestamp("2026-10-04T12:00:00Z")
    out = _parse_weather_payload(
        {"hourly": {"time": ["2026-10-04T12:00:00Z"], "temperature_2m": [18]}},
        kickoff,
    )
    assert "rich_weather_relative_humidity_2m" not in out
    assert "rich_weather_wind_gusts_10m" not in out


def test_h2h_summary_respects_team_orientation_and_cutoff():
    kickoff = pd.Timestamp("2026-10-04T12:00:00Z")
    events = [
        {
            "startTimestamp": int(pd.Timestamp("2026-10-03T12:00:00Z").timestamp()),
            "homeTeam": {"id": 2},
            "awayTeam": {"id": 1},
            "homeScore": {"current": 1},
            "awayScore": {"current": 2},
        },
        {
            "startTimestamp": int(pd.Timestamp("2026-10-05T12:00:00Z").timestamp()),
            "homeTeam": {"id": 1},
            "awayTeam": {"id": 2},
            "homeScore": {"current": 9},
            "awayScore": {"current": 0},
        },
    ]
    out = _summarize_h2h(events, "1", "2", kickoff)
    assert out["rich_h2h_matches"] == 1
    assert out["rich_h2h_home_wins"] == 1
    assert out["rich_h2h_away_goals"] == 1
    assert out["rich_h2h_home_goals"] == 2
    assert out["rich_h2h_last5_scorelines"] == "2-1"


def test_rich_workflow_outputs_are_research_only():
    from pathlib import Path

    workflow = Path(
        ".github/workflows/soccer-matchday-intelligence.yml"
    ).read_text(encoding="utf-8")
    assert "matchday_rich_enrichment.csv" in workflow
    assert "matchday_rich_raw.jsonl" in workflow
    assert "production_changed" in workflow
