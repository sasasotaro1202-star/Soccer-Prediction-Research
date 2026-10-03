from __future__ import annotations

from types import SimpleNamespace

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


def test_live_availability_smoke_uses_bounded_fixture_scan():
    from pathlib import Path

    workflow = Path(".github/workflows/soccer-rich-data-availability.yml").read_text(encoding="utf-8")
    assert "--horizon-hours 168" in workflow
    assert "--max-events 12" in workflow
    assert "production collection keeps its uncapped discovery" in workflow


def test_market_summary_keeps_optional_total_spread_and_timestamp():
    payload = {
        "odds": [
            {
                "provider": {"name": "A", "priority": 1},
                "homeTeamOdds": {"decimalValue": 2.0},
                "drawOdds": {"decimalValue": 3.5},
                "awayTeamOdds": {"decimalValue": 4.0},
                "spread": -0.5,
                "overUnder": 2.75,
                "lastUpdated": "2026-10-04T10:00:00Z",
            }
        ]
    }
    out = _market_summary([payload])
    assert out["rich_market_spread_mean"] == -0.5
    assert out["rich_market_total_mean"] == 2.75
    assert out["rich_market_latest_observation_at_utc"] == "2026-10-04T10:00:00+00:00"
    assert out["rich_market_observation_timestamp_count"] == 1


def test_lineup_parser_keeps_player_composition_without_turning_missing_into_zero():
    from src.data.matchday_rich_enrichment import _parse_lineups

    payload = {
        "confirmed": True,
        "home": {
            "formation": "4-3-3",
            "players": [
                {
                    "starter": True,
                    "captain": True,
                    "player": {
                        "id": 10,
                        "name": "A",
                        "position": {"shortName": "D"},
                        "age": 28,
                        "height": 190,
                        "preferredFoot": "Right",
                        "marketValue": 25,
                    },
                },
                {
                    "substitute": True,
                    "player": {
                        "id": 11,
                        "name": "B",
                        "position": {"shortName": "F"},
                    },
                },
            ],
            "missingPlayers": [
                {"player": {"id": 12, "name": "C"}, "reason": "Injury"}
            ],
        },
        "away": {},
    }
    out = _parse_lineups(payload)
    assert out["rich_sofa_home_starter_names"] == "A"
    assert out["rich_sofa_home_starter_positions"] == "D"
    assert out["rich_sofa_home_starter_position_count_d"] == 1
    assert out["rich_sofa_home_starter_avg_age"] == 28.0
    assert out["rich_sofa_home_starter_avg_height_cm"] == 190.0
    assert out["rich_sofa_home_starter_market_value_sum"] == 25.0
    assert out["rich_sofa_home_missing_ids"] == "12"
    assert out["rich_sofa_home_missing_names"] == "C"
    assert out["rich_sofa_home_missing_reasons"] == "Injury"
    assert out["rich_sofa_lineup_confirmed"] is True
    assert pd.isna(out["rich_sofa_away_starter_count"])

def test_standings_parser_extracts_total_home_and_away_context():
    from src.data.matchday_rich_enrichment import _parse_sofa_standings

    payload = {
        "standings": [{
            "rows": [
                {
                    "team": {"id": 10, "name": "Home"},
                    "position": 3, "matches": 8, "wins": 5, "draws": 2, "losses": 1,
                    "scoresFor": 15, "scoresAgainst": 7, "points": 17,
                    "scoreDiffFormatted": "+8",
                    "form": ["W", "W", "D"],
                },
                {
                    "team": {"id": 20, "name": "Away"},
                    "position": 9, "matches": 8, "wins": 3, "draws": 2, "losses": 3,
                    "scoresFor": 11, "scoresAgainst": 10, "points": 11,
                    "scoreDiffFormatted": "+1",
                },
            ]
        }]
    }
    out = _parse_sofa_standings(payload, "10", "20", "total")
    assert out["rich_sofa_standing_total_home_position"] == 3
    assert out["rich_sofa_standing_total_home_points"] == 17
    assert out["rich_sofa_standing_total_home_score_diff_formatted"] == "+8"
    assert out["rich_sofa_standing_total_home_form"] == "W|W|D"
    assert out["rich_sofa_standing_total_away_position"] == 9
    assert out["rich_sofa_standing_total_away_goals_for"] == 11


def test_standings_parser_does_not_fill_missing_team_as_zero():
    from src.data.matchday_rich_enrichment import _parse_sofa_standings

    payload = {"standings": [{"rows": [{"team": {"id": 10}, "position": 3, "points": 17}]}]}
    out = _parse_sofa_standings(payload, "10", "20", "home")
    assert out["rich_sofa_standing_home_home_position"] == 3
    assert "rich_sofa_standing_home_away_position" not in out

def test_injury_parser_retains_detail_without_coercing_unknowns():
    from src.data.matchday_rich_enrichment import _parse_injury_details

    out = _parse_injury_details({
        "injuries": [
            {
                "athlete": {"displayName": "Player A"},
                "status": "Out",
                "reason": "Hamstring",
                "returnDate": "2026-10-10",
            },
            {
                "playerName": "Player B",
                "status": "Questionable",
            },
        ]
    })
    assert out["count"] == 2
    assert out["severe_count"] == 1
    assert out["names"] == "Player A|Player B"
    assert out["statuses"] == "Out|Questionable"
    assert out["reasons"] == "Hamstring"
    assert out["expected_return"] == "2026-10-10"

def test_source_coverage_report_counts_real_payloads_and_feature_channels():
    from src.data.matchday_rich_enrichment import _source_coverage_report

    frame = pd.DataFrame([
        {
            "match_id": "m1",
            "rich_detail_status": "ENRICHED",
            "rich_market_provider_count": 2,
            "rich_sofa_home_formation": "4-3-3",
            "rich_espn_home_injury_count": 1,
            "rich_sofa_standing_total_home_position": 2,
            "rich_weather_nearest_valid_time_utc": "2026-10-04T12:00:00Z",
            "rich_h2h_matches": 4,
            "rich_recent_home_matches": 5,
        },
        {
            "match_id": "m2",
            "rich_detail_status": "ENRICHED",
            "rich_market_provider_count": 1,
            "rich_sofa_home_formation": "4-4-2",
            "rich_espn_home_injury_count": 0,
            "rich_sofa_standing_total_home_position": 8,
            "rich_weather_nearest_valid_time_utc": "",
            "rich_h2h_matches": float("nan"),
            "rich_recent_home_matches": 5,
        },
    ])
    raw = [
        {"match_id": "m1", "provider_family": "espn", "endpoint": "summary"},
        {"match_id": "m1", "provider_family": "sofascore", "endpoint": "event"},
        {"match_id": "m1", "provider_family": "open_meteo", "endpoint": "forecast"},
        {"match_id": "m2", "provider_family": "espn", "endpoint": "summary"},
    ]
    report = _source_coverage_report(frame, raw)
    assert report["acquisition_state"] == "PAYLOADS_OBSERVED"
    assert report["source_families"]["espn"]["unique_match_ids"] == 2
    assert report["source_families"]["sofascore"]["match_coverage_pct"] == 50.0
    assert report["feature_channels"]["market"]["rows_with_data"] == 2
    assert report["feature_channels"]["injury"]["rows_with_data"] == 2
    assert report["feature_channels"]["weather"]["rows_with_data"] == 1
    assert report["feature_channels"]["h2h"]["rows_with_data"] == 1

def test_parse_fotmob_matches_filters_future_supported_competitions():
    from src.data.matchday_intelligence_fetch import parse_fotmob_matches

    payload = {
        "leagues": [
            {
                "id": 47,
                "name": "Premier League",
                "matches": [{
                    "id": 12345,
                    "status": {"utcTime": "2026-10-05T15:00:00.000Z"},
                    "home": {"id": 1, "name": "Home FC"},
                    "away": {"id": 2, "name": "Away FC"},
                }],
            },
            {
                "id": 999,
                "name": "Unsupported League",
                "matches": [{
                    "id": 9,
                    "status": {"utcTime": "2026-10-05T16:00:00.000Z"},
                    "home": {"id": 3, "name": "X"},
                    "away": {"id": 4, "name": "Y"},
                }],
            },
        ]
    }
    rows = parse_fotmob_matches(
        payload,
        now=pd.Timestamp("2026-10-04T00:00:00Z"),
        horizon_hours=48,
        available_at="2026-10-04T00:05:00Z",
    )
    assert len(rows) == 1
    assert rows[0]["competition"] == "EPL"
    assert rows[0]["fotmob_match_id"] == "12345"
    assert rows[0]["home_team"] == "Home FC"


def test_parse_fotmob_detail_preserves_pre_match_lineup_and_never_requires_missing_values():
    from src.data.matchday_rich_enrichment import _parse_fotmob_detail

    payload = {
        "general": {
            "matchId": "12345",
            "matchRound": "7",
            "leagueName": "Premier League",
            "countryCode": "ENG",
            "coverageLevel": "xG",
            "started": False,
            "finished": False,
            "homeTeam": {"id": 1, "name": "Home FC"},
            "awayTeam": {"id": 2, "name": "Away FC"},
        },
        "header": {
            "status": {"finished": False, "started": False, "cancelled": False, "reason": {"long": "Not started"}},
            "teams": [{"id": 1, "name": "Home FC"}, {"id": 2, "name": "Away FC"}],
        },
        "content": {
            "lineup": {
                "homeTeam": {
                    "id": 1,
                    "name": "Home FC",
                    "formation": "4-3-3",
                    "averageStarterAge": 27.5,
                    "totalStarterMarketValue": 100000000,
                    "starters": [{"id": 101, "name": "Player A"}],
                    "unavailable": [{"id": 102}],
                }
            }
        },
    }
    out = _parse_fotmob_detail(payload)
    assert out["rich_fotmob_match_id"] == "12345"
    assert out["rich_fotmob_started"] is False
    assert out["rich_fotmob_home_formation"] == "4-3-3"
    assert out["rich_fotmob_home_starter_count"] == 1
    assert out["rich_fotmob_home_unavailable_count"] == 1
    assert "rich_fotmob_prestats_xg_home" not in out


def test_fotmob_json_fetch_falls_back_to_current_data_route():
    from src.data.matchday_intelligence_fetch import _get_json

    class FakeFetcher:
        def __init__(self):
            self.urls = []

        def get(self, source, url, **kwargs):
            self.urls.append(url)
            if url == "https://www.fotmob.com/api/matches":
                raise RuntimeError("primary route unavailable")
            return SimpleNamespace(
                body=b'{"leagues": []}',
                metadata=SimpleNamespace(retrieved_at="2026-10-04T00:00:00Z"),
            )

    fetcher = FakeFetcher()
    payload, retrieved_at = _get_json(
        fetcher,
        "fotmob_matches",
        "https://www.fotmob.com/api/matches",
        {"date": "20261004"},
    )
    assert payload == {"leagues": []}
    assert retrieved_at == "2026-10-04T00:00:00Z"
    assert fetcher.urls == [
        "https://www.fotmob.com/api/matches",
        "https://www.fotmob.com/api/data/matches",
    ]


def test_fotmob_detail_json_fetch_falls_back_to_current_data_route():
    from src.data.matchday_intelligence_fetch import _get_json

    class FakeFetcher:
        def __init__(self):
            self.urls = []

        def get(self, source, url, **kwargs):
            self.urls.append(url)
            if url == "https://www.fotmob.com/api/matchDetails":
                raise RuntimeError("primary detail route unavailable")
            return SimpleNamespace(
                body=b'{"general": {"matchId": "12345"}}',
                metadata=SimpleNamespace(retrieved_at="2026-10-04T00:01:00Z"),
            )

    fetcher = FakeFetcher()
    payload, retrieved_at = _get_json(
        fetcher,
        "fotmob_match_detail",
        "https://www.fotmob.com/api/matchDetails",
        {"matchId": "12345"},
    )
    assert payload["general"]["matchId"] == "12345"
    assert retrieved_at == "2026-10-04T00:01:00Z"
    assert fetcher.urls == [
        "https://www.fotmob.com/api/matchDetails",
        "https://www.fotmob.com/api/data/matchDetails",
    ]


def test_fotmob_identity_does_not_alias_other_countries_to_same_competition_name():
    from src.data.matchday_intelligence_fetch import parse_fotmob_matches

    payload = {
        "leagues": [
            {
                "id": 522,
                "ccode": "GHA",
                "name": "Premier League",
                "matches": [{
                    "id": 5975044,
                    "status": {"utcTime": "2026-10-04T15:00:00.000Z"},
                    "home": {"id": 102019, "name": "Heart of Lions"},
                    "away": {"id": 102017, "name": "Ashanti Gold"},
                }],
            },
            {
                "id": 47,
                "ccode": "ENG",
                "name": "Premier League",
                "matches": [{
                    "id": 5795427,
                    "status": {"utcTime": "2026-10-05T15:00:00.000Z"},
                    "home": {"id": 8455, "name": "Chelsea"},
                    "away": {"id": 10204, "name": "Brighton"},
                }],
            },
        ]
    }
    rows = parse_fotmob_matches(
        payload,
        now=pd.Timestamp("2026-10-04T00:00:00Z"),
        horizon_hours=48,
        available_at="2026-10-04T00:05:00Z",
    )
    assert len(rows) == 1
    assert rows[0]["competition"] == "EPL"
    assert rows[0]["fotmob_league_id"] == "47"
    assert rows[0]["home_team"] == "Chelsea"


def test_safe_provider_id_removes_csv_float_suffix():
    from src.data.matchday_rich_enrichment import _safe_provider_id

    assert _safe_provider_id("5975044.0") == "5975044"
    assert _safe_provider_id(5975044) == "5975044"


def test_fotmob_detail_validation_rejects_error_payload_and_mismatch():
    from src.data.matchday_rich_enrichment import _validate_fotmob_detail_payload

    with_error = {"error": True, "matchId": "5975044", "message": "Data not found"}
    try:
        _validate_fotmob_detail_payload(with_error, "5975044")
    except RuntimeError as exc:
        assert "Data not found" in str(exc)
    else:
        raise AssertionError("error payload must be rejected")

    mismatched = {"general": {"matchId": "999"}}
    try:
        _validate_fotmob_detail_payload(mismatched, "5975044")
    except RuntimeError as exc:
        assert "matchId mismatch" in str(exc)
    else:
        raise AssertionError("mismatched payload must be rejected")


def test_matchday_fetch_has_integer_normalizer_for_provider_ids():
    from src.data.matchday_intelligence_fetch import _safe_int

    assert _safe_int("47") == 47
    assert _safe_int("47.0") == 47
    assert _safe_int("not-an-int") is None
