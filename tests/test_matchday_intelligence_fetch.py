from datetime import datetime, timezone
import pandas as pd

from src.data.matchday_intelligence_fetch import (
    ESPN_LEAGUES,
    FOOTBALL_DATA_DIVISIONS,
    _devig,
    parse_football_data_fixtures,
    parse_sofascore_event,
    _sofascore_competition,
    parse_event_roster,
    parse_injury_impact,
    parse_market_odds,
    _get_json,
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


def test_weather_missing_fields_remain_unknown():
    kickoff = pd.Timestamp("2026-09-25T12:00:00Z")
    severity, temp = parse_weather_severity(
        {"hourly": {"time": ["2026-09-25T12:00:00Z"], "temperature_2m": [20]}},
        kickoff,
    )
    assert pd.isna(severity)
    assert pd.isna(temp)


def test_live_scope_maps_requested_uefa_and_friendly_competitions():
    assert _sofascore_competition({"tournament": {"name": "UEFA Conference League"}}) == "UECL"
    assert _sofascore_competition({"tournament": {"name": "UEFA Nations League"}}) == "UEFA_NATIONS_LEAGUE_M"
    assert _sofascore_competition({"tournament": {"name": "Emperor's Cup"}}) == "EMP_CUP"
    assert _sofascore_competition({"tournament": {"name": "International Friendly Games"}}) == "FRIENDLY"


def test_matchday_base_row_preserves_unknown_optional_signals():
    from src.data.matchday_intelligence_fetch import _matchday_base_row

    row = _matchday_base_row(
        match_id="m1",
        kickoff=pd.Timestamp("2026-09-25T12:00:00Z"),
        home_team="A",
        away_team="B",
        competition="EPL",
        source="test",
        available_at="2026-09-24T12:00:00Z",
    )
    for key in (
        "matchday_injury_impact_home",
        "matchday_injury_impact_away",
        "matchday_lineup_impact_home",
        "matchday_lineup_impact_away",
        "matchday_weather_penalty_home",
        "matchday_weather_penalty_away",
        "matchday_rest_diff_hours",
    ):
        assert pd.isna(row[key])
    assert pd.isna(row["source_available_at_utc"])
    assert row["pit_verified"] is False
    assert row["source_retrieved_at_utc"] == "2026-09-24T12:00:00Z"
    assert pd.isna(row["matchday_available_at_utc"])
    assert row["matchday_pit_verified"] is False
    assert row["matchday_retrieved_at_utc"] == "2026-09-24T12:00:00Z"


def test_asian_games_are_not_misrouted_to_argentina():
    assert "AG_M" not in ESPN_LEAGUES
    assert FOOTBALL_DATA_DIVISIONS.get("ARG") == "ARG"
    assert _sofascore_competition({"tournament": {"name": "Asian Games"}}) == "AG_M"
    assert _sofascore_competition({"tournament": {"name": "Asian Games, Women"}}) == "AG_W"
    assert _sofascore_competition({"tournament": {"name": "Liga Profesional de Fútbol"}}) == "ARG"


def test_weather_without_timestamps_remains_unknown():
    severity, temp = parse_weather_severity({"hourly": {}}, pd.Timestamp("2026-09-25T12:00:00Z"))
    assert pd.isna(severity)
    assert pd.isna(temp)


def test_sofascore_uses_public_api_host():
    from src.data import matchday_intelligence_fetch as m
    source = m.SOFASCORE_COMPETITIONS
    assert source["Premier League"] == "EPL"
    import inspect
    code = inspect.getsource(m._collect_sofascore_day)
    lineup_code = inspect.getsource(m._enrich_sofascore_lineup)
    assert "https://api.sofascore.com/api/v1/sport/football/scheduled-events/" in code
    assert "https://api.sofascore.com/api/v1/event/{event_id}/lineups" in lineup_code


def test_sofascore_prefers_canonical_unique_tournament_for_qualifiers():
    payload = {
        "tournament": {"name": "UEFA Champions League, Qualification"},
        "uniqueTournament": {"name": "UEFA Champions League"},
    }
    assert _sofascore_competition(payload) == "UCL"


def test_espn_json_falls_back_to_web_hostname_after_html_response():
    from types import SimpleNamespace

    class FakeFetcher:
        def __init__(self):
            self.calls = []

        def get(self, source, url, **kwargs):
            self.calls.append(url)
            body = (
                b"<!doctype html><html><body>blocked</body></html>"
                if len(self.calls) == 1
                else b'{"events": []}'
            )
            return SimpleNamespace(
                body=body,
                metadata=SimpleNamespace(retrieved_at="2026-09-25T12:00:00Z"),
            )

    fetcher = FakeFetcher()
    payload, retrieved_at = _get_json(
        fetcher,
        "espn_scoreboard",
        "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard",
        {"dates": "20260925"},
    )
    assert payload == {"events": []}
    assert retrieved_at == "2026-09-25T12:00:00Z"
    assert fetcher.calls == [
        "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard",
        "https://site.web.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard",
    ]


def test_football_data_complements_partial_sofascore_coverage(monkeypatch):
    import src.data.matchday_intelligence_fetch as m

    class DummyFetcher:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, *args, **kwargs):
            raise RuntimeError("ESPN unavailable")

    monkeypatch.setattr(m, "ExternalFetcher", DummyFetcher)
    monkeypatch.setattr(
        m,
        "_now",
        lambda: datetime(2026, 9, 25, 0, 0, tzinfo=timezone.utc),
    )
    base_time = pd.Timestamp("2026-09-25T00:00:00Z")
    sofa = m._matchday_base_row(
        match_id="sofa:partial",
        kickoff=base_time + pd.Timedelta(hours=6),
        home_team="Sofa Home",
        away_team="Sofa Away",
        competition="EPL",
        source="sofascore",
        available_at="2026-09-25T00:01:00Z",
    )
    football_data = m._matchday_base_row(
        match_id="fdx:complement",
        kickoff=base_time + pd.Timedelta(hours=12),
        home_team="FD Home",
        away_team="FD Away",
        competition="EPL",
        source="football-data.co.uk",
        available_at="2026-09-25T00:02:00Z",
    )
    monkeypatch.setattr(
        m,
        "_collect_sofascore_day",
        lambda *args, **kwargs: ([sofa], [], "2026-09-25T00:01:00Z"),
    )
    monkeypatch.setattr(
        m,
        "_collect_football_data_fallback",
        lambda *args, **kwargs: ([football_data], [], "2026-09-25T00:02:00Z"),
    )

    frame, status = m.collect_matchday_snapshots(days=1, horizon_hours=24, max_events=20)
    assert set(frame["match_id"]) == {"sofa:partial", "fdx:complement"}
    assert any(x["provider"] == "sofascore" for x in status["fallback_usage"])
    assert any(x["provider"] == "football-data.co.uk" for x in status["fallback_usage"])

def test_matchday_lineup_signal_uses_confirmed_availability_burden_not_fixed_zero_point_five():
    import src.data.matchday_intelligence_fetch as m
    row = m._matchday_base_row(
        match_id="m-lineup",
        kickoff=pd.Timestamp("2026-09-25T12:00:00Z"),
        home_team="A",
        away_team="B",
        competition="EPL",
        source="sofascore",
        available_at="2026-09-25T10:00:00Z",
    )
    row["sofascore_event_id"] = "12345"
    class Dummy:
        pass
    class Fetcher:
        def get(self, *args, **kwargs):
            return Dummy()
    # Patch the JSON helper because _enrich_sofascore_lineup only needs its decoded payload.
    original = m._get_json
    try:
        m._get_json = lambda *args, **kwargs: ({
            "confirmed": True,
            "home": {
                "players": [{"starter": True, "player": {"id": str(i)}} for i in range(11)],
                "missingPlayers": [{}],
            },
            "away": {
                "players": [{"starter": True, "player": {"id": str(i)}} for i in range(11, 22)],
                "missingPlayers": [],
            },
        }, "2026-09-25T11:00:00Z")
        updated, _ = m._enrich_sofascore_lineup(Fetcher(), row)
    finally:
        m._get_json = original
    assert updated["starter_status"] == "ANNOUNCED"
    assert updated["matchday_lineup_missing_count_home"] == 1
    assert updated["matchday_lineup_missing_count_away"] == 0
    assert updated["matchday_lineup_impact_home"] > updated["matchday_lineup_impact_away"]
    assert updated["matchday_lineup_impact_away"] == 0.50


def test_retrieval_time_is_never_promoted_to_source_availability():
    from src.data.matchday_intelligence_fetch import _matchday_base_row

    row = _matchday_base_row(
        match_id="retrieval-only",
        kickoff=pd.Timestamp("2026-09-25T12:00:00Z"),
        home_team="A",
        away_team="B",
        competition="EPL",
        source="espn_scoreboard",
        available_at="2026-09-25T08:00:00Z",
    )
    assert pd.isna(row["source_available_at_utc"])
    assert row["pit_verified"] is False
    assert pd.isna(row["matchday_available_at_utc"])
    assert row["matchday_pit_verified"] is False
    assert row["source_retrieved_at_utc"] == "2026-09-25T08:00:00Z"
    assert row["matchday_retrieved_at_utc"] == "2026-09-25T08:00:00Z"
