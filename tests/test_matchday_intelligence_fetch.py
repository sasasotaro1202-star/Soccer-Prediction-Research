import pandas as pd

from src.data.matchday_intelligence_fetch import (
    _devig,
    parse_event_roster,
    parse_injury_impact,
    parse_market_odds,
    parse_weather_severity,
)


def test_devig_is_normalized_and_ordered():
    probs = _devig((2.0, 3.6, 4.2))
    assert abs(sum(probs) - 1.0) < 1e-12
    assert probs[0] > probs[1] > probs[2]


def test_parse_market_odds_selects_lowest_priority_provider():
    payload = {
        "odds": [
            {
                "provider": {"name": "later", "priority": 5},
                "homeTeamOdds": {"value": 2.4},
                "drawOdds": {"value": 3.5},
                "awayTeamOdds": {"value": 3.0},
            },
            {
                "provider": {"name": "primary", "priority": 1},
                "homeTeamOdds": {"value": 2.0},
                "drawOdds": {"value": 3.6},
                "awayTeamOdds": {"value": 4.2},
            },
        ]
    }
    odds, probs, provider = parse_market_odds(payload)
    assert odds == (2.0, 3.6, 4.2)
    assert provider == "primary"
    assert abs(sum(probs) - 1.0) < 1e-12


def test_parse_injury_impact_is_capped_and_weighted():
    payload = {
        "injuries": [
            {"status": "Out"},
            {"status": "Doubtful"},
            {"status": "Questionable"},
        ]
    }
    impact, severe, confidence = parse_injury_impact(payload)
    assert 0.0 < impact <= 1.0
    assert severe == 2
    assert 0.5 <= confidence <= 0.95


def test_event_roster_requires_explicit_starter_flags():
    payload = {
        "entries": [{"starter": True, "playerId": str(i)} for i in range(11)]
        + [{"starter": False, "playerId": "bench"}]
    }
    count, ids = parse_event_roster(payload)
    assert count == 11
    assert len(ids) == 11


def test_weather_severity_is_bounded():
    kickoff = pd.Timestamp("2026-09-25T12:00:00Z")
    payload = {
        "hourly": {
            "time": ["2026-09-25T12:00:00Z"],
            "precipitation_probability": [90],
            "windspeed_10m": [40],
            "temperature_2m": [5],
            "weathercode": [95],
        }
    }
    severity, temp = parse_weather_severity(payload, kickoff)
    assert 0.0 <= severity <= 1.0
    assert temp == 5.0
