import pandas as pd

from src.data.matchday_intelligence_fetch import (
    _devig,
    parse_football_data_fixtures,
    parse_sofascore_event,
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


def test_parse_market_odds_prefers_explicit_decimal_value():
    payload = {
        "odds": [{
            "provider": {"name": "explicit-decimal", "priority": 1},
            "homeTeamOdds": {"value": 250, "decimalValue": 3.50},
            "drawOdds": {"value": 300, "decimalValue": 3.00},
            "awayTeamOdds": {"value": 225, "decimalValue": 2.25},
        }]
    }
    odds, probs, provider = parse_market_odds(payload)
    assert odds == (3.50, 3.00, 2.25)
    assert provider == "explicit-decimal"
    assert abs(sum(probs) - 1.0) < 1e-12


def test_parse_sofascore_event_maps_supported_tournament():
    payload = {
        "id": 123,
        "startTimestamp": 1790337600,
        "tournament": {"name": "Premier League"},
        "homeTeam": {"id": 1, "name": "Home FC"},
        "awayTeam": {"id": 2, "name": "Away FC"},
        "venue": {"name": "Test Stadium", "city": "London", "country": {"name": "England"}},
    }
    row = parse_sofascore_event(payload)
    assert row is not None
    assert row["competition"] == "EPL"
    assert row["home_team"] == "Home FC"
    assert row["away_team"] == "Away FC"


def test_parse_football_data_fixtures_uses_current_fixture_shape():
    frame = pd.DataFrame([{
        "Div": "E0",
        "Date": "25/09/2026",
        "Time": "20:00",
        "HomeTeam": "Home FC",
        "AwayTeam": "Away FC",
        "AvgH": 2.0,
        "AvgD": 3.5,
        "AvgA": 4.0,
    }])
    rows = parse_football_data_fixtures(
        frame,
        now=pd.Timestamp("2026-09-25T00:00:00Z"),
        horizon_hours=24,
        available_at="2026-09-25T00:05:00Z",
        max_events=10,
    )
    assert len(rows) == 1
    assert rows[0]["competition"] == "EPL"
    assert rows[0]["matchday_market_provider"] == "football-data:average"
    probs = [rows[0]["matchday_market_p_home"], rows[0]["matchday_market_p_draw"], rows[0]["matchday_market_p_away"]]
    assert abs(sum(probs) - 1.0) < 1e-12


def test_football_data_fixture_with_unknown_division_is_ignored():
    frame = pd.DataFrame([{
        "Div": "UNKNOWN",
        "Date": "25/09/2026",
        "Time": "20:00",
        "HomeTeam": "Home FC",
        "AwayTeam": "Away FC",
    }])
    rows = parse_football_data_fixtures(
        frame,
        now=pd.Timestamp("2026-09-25T00:00:00Z"),
        horizon_hours=24,
        available_at="2026-09-25T00:05:00Z",
        max_events=10,
    )
    assert rows == []
