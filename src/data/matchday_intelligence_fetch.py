from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from io import BytesIO
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.external_fetch import ExternalFetcher, iso_utc


ESPN_LEAGUES: dict[str, str] = {
    "EPL": "eng.1",
    "ERE": "ned.1",
    "LL": "esp.1",
    "SA": "ita.1",
    "BL1": "ger.1",
    "FL1": "fra.1",
    "J1": "jpn.1",
    "J2": "jpn.2",
    "J3": "jpn.3",
    "UCL": "uefa.champions",
    "UEL": "uefa.europa",
    "AG_M": "arg.1",
    "MLS": "usa.1",
}

DETAILED_HORIZON_HOURS = 12.0
MAX_EVENTS = 80
SOFASCORE_LINEUP_ENRICH_LIMIT = 16
FOOTBALL_DATA_FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

SOFASCORE_COMPETITIONS: dict[str, str] = {
    "Premier League": "EPL",
    "Eredivisie": "ERE",
    "LaLiga": "LL",
    "La Liga": "LL",
    "Serie A": "SA",
    "Bundesliga": "BL1",
    "Ligue 1": "FL1",
    "J1 League": "J1",
    "J2 League": "J2",
    "J3 League": "J3",
    "Emperor's Cup": "EMP_CUP",
    "Emperor’s Cup": "EMP_CUP",
    "Japan FA Cup": "EMP_CUP",
    "UEFA Champions League": "UCL",
    "UEFA Europa League": "UEL",
    "UEFA Conference League": "UECL",
    "UEFA Super Cup": "UEFA_SUPER_CUP",
    "UEFA Youth League": "UEFA_YOUTH_LEAGUE",
    "UEFA Women's Champions League": "UWCL",
    "UEFA Women’s Champions League": "UWCL",
    "UEFA Women's Europa Cup": "UWEC",
    "UEFA Women’s Europa Cup": "UWEC",
    "UEFA European Championship": "UEFA_EURO_M",
    "UEFA European Championship (EURO)": "UEFA_EURO_M",
    "UEFA European Qualifiers": "UEFA_EURO_QUALI_M",
    "UEFA Nations League": "UEFA_NATIONS_LEAGUE_M",
    "UEFA Women's European Championship": "UEFA_EURO_W",
    "UEFA Women’s European Championship": "UEFA_EURO_W",
    "UEFA Women's European Qualifiers": "UEFA_EURO_QUALI_W",
    "UEFA Women’s European Qualifiers": "UEFA_EURO_QUALI_W",
    "UEFA Women's Nations League": "UEFA_NATIONS_LEAGUE_W",
    "UEFA Women’s Nations League": "UEFA_NATIONS_LEAGUE_W",
    "UEFA European Under-21 Championship": "UEFA_U21",
    "UEFA European Under-19 Championship": "UEFA_U19",
    "UEFA European Under-17 Championship": "UEFA_U17",
    "UEFA Women's European Under-19 Championship": "UEFA_WU19",
    "UEFA Women’s European Under-19 Championship": "UEFA_WU19",
    "UEFA Women's European Under-17 Championship": "UEFA_WU17",
    "UEFA Women’s European Under-17 Championship": "UEFA_WU17",
    "UEFA Regions' Cup": "UEFA_REGIONS_CUP",
    "UEFA Regions’ Cup": "UEFA_REGIONS_CUP",
    "International Friendly": "FRIENDLY",
    "International Friendlies": "FRIENDLY",
    "International Friendly Games": "FRIENDLY",
    "Club Friendly": "FRIENDLY",
    "Club Friendly Games": "FRIENDLY",
    "Friendlies": "FRIENDLY",
    "Major League Soccer": "MLS",
    "Liga Profesional de Fútbol": "AG_M",
    "Liga Profesional": "AG_M",
}

FOOTBALL_DATA_DIVISIONS: dict[str, str] = {
    "E0": "EPL",
    "N1": "ERE",
    "SP1": "LL",
    "I1": "SA",
    "D1": "BL1",
    "F1": "FL1",
    "J1": "J1",
    "J2": "J2",
    "USA": "MLS",
    "ARG": "AG_M",
}
FOOTBALL_DATA_TZ: dict[str, str] = {
    "EPL": "Europe/London",
    "ERE": "Europe/Amsterdam",
    "LL": "Europe/Madrid",
    "SA": "Europe/Rome",
    "BL1": "Europe/Berlin",
    "FL1": "Europe/Paris",
    "J1": "Asia/Tokyo",
    "J2": "Asia/Tokyo",
    "MLS": "America/New_York",
    "AG_M": "America/Argentina/Buenos_Aires",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ts(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    out = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(out) else pd.Timestamp(out)


def _number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _american_prob(value: Any) -> float | None:
    number = _number(value)
    if number is None or number == 0:
        return None
    return 100.0 / (number + 100.0) if number > 0 else -number / (-number + 100.0)


def _devig(odds: tuple[float, float, float]) -> tuple[float, float, float]:
    inv = np.asarray([1.0 / max(float(v), 1.000001) for v in odds], dtype=float)
    total = float(inv.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("invalid market odds")
    inv /= total
    return tuple(float(v) for v in inv)


def _get_csv(
    fetcher: ExternalFetcher,
    source: str,
    url: str,
) -> tuple[pd.DataFrame, str]:
    response = fetcher.get(
        source,
        url,
        headers={"User-Agent": "Soccer-Prediction-Research/1.0"},
        cache_ttl_seconds=900.0,
    )
    try:
        frame = pd.read_csv(BytesIO(response.body), encoding="utf-8", encoding_errors="replace")
    except UnicodeDecodeError:
        frame = pd.read_csv(BytesIO(response.body), encoding="cp1252", encoding_errors="replace")
    except Exception as exc:
        raise RuntimeError(f"{source}: invalid CSV") from exc
    return frame, response.metadata.retrieved_at


def _get_json(
    fetcher: ExternalFetcher,
    source: str,
    url: str,
    params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    response = fetcher.get(
        source,
        url,
        params=params,
        headers={"User-Agent": "Soccer-Prediction-Research/1.0"},
        cache_ttl_seconds=600.0,
    )
    try:
        payload = json.loads(response.body.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{source}: invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{source}: payload root is not an object")
    return payload, response.metadata.retrieved_at


def parse_market_odds(
    summary: dict[str, Any],
) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None, str]:
    candidates: list[tuple[int, str, tuple[float, float, float]]] = []
    for item in summary.get("odds", []) or []:
        if not isinstance(item, dict):
            continue
        home_obj = item.get("homeTeamOdds") or {}
        away_obj = item.get("awayTeamOdds") or {}
        draw_obj = item.get("drawOdds") or item.get("drawTeamOdds") or {}
        home = _number(home_obj.get("decimalValue"))
        draw = _number(draw_obj.get("decimalValue"))
        away = _number(away_obj.get("decimalValue"))
        if home is None:
            home = _number(home_obj.get("value"))
        if draw is None:
            draw = _number(draw_obj.get("value"))
        if away is None:
            away = _number(away_obj.get("value"))
        provider = str((item.get("provider") or {}).get("name") or "ESPN")
        priority = int(_number((item.get("provider") or {}).get("priority")) or 9999)
        if home is not None and draw is not None and away is not None and min(home, draw, away) > 1.0:
            candidates.append((priority, provider, (home, draw, away)))
            continue
        hp = _american_prob(home_obj.get("moneyLine"))
        dp = _american_prob(draw_obj.get("moneyLine"))
        ap = _american_prob(away_obj.get("moneyLine"))
        if hp is not None and dp is not None and ap is not None:
            values = np.asarray([hp, dp, ap], dtype=float)
            values /= values.sum()
            return None, tuple(float(v) for v in values), provider
    if not candidates:
        return None, None, ""
    _, provider, odds = sorted(candidates, key=lambda x: (x[0], x[1]))[0]
    return odds, _devig(odds), provider


def parse_injury_impact(payload: dict[str, Any]) -> tuple[float, int, float]:
    weights = {
        "out": 1.0,
        "doubtful": 0.70,
        "questionable": 0.35,
        "day-to-day": 0.35,
        "probable": 0.10,
    }
    total = 0.0
    severe = 0
    entries = payload.get("injuries", []) or []
    for item in entries:
        if not isinstance(item, dict):
            continue
        status = str(
            item.get("status")
            or (item.get("fantasy") or {}).get("status")
            or ""
        ).strip().lower()
        weight = next((v for k, v in weights.items() if k in status), 0.0)
        total += weight
        if weight >= 0.70:
            severe += 1
    impact = float(np.clip(total / 4.0, 0.0, 1.0))
    confidence = float(np.clip(0.50 + min(len(entries), 8) * 0.06, 0.50, 0.95))
    return impact, severe, confidence


def parse_event_roster(payload: dict[str, Any]) -> tuple[int, set[str]]:
    starters: set[str] = set()
    for item in payload.get("entries", []) or []:
        if not isinstance(item, dict) or item.get("starter") is not True:
            continue
        athlete = item.get("athlete") or {}
        athlete_id = athlete.get("id") or item.get("playerId")
        if athlete_id is not None:
            starters.add(str(athlete_id))
    return len(starters), starters


def parse_weather_severity(
    payload: dict[str, Any],
    kickoff: pd.Timestamp,
) -> tuple[float, float]:
    hourly = payload.get("hourly") or {}
    times = pd.to_datetime(pd.Series(hourly.get("time") or []), utc=True, errors="coerce")
    if times.empty or times.isna().all():
        return 0.0, 0.0
    index = int(np.argmin(np.abs((times - kickoff).dt.total_seconds().to_numpy(dtype=float))))

    def value(key: str, default: float = 0.0) -> float:
        values = hourly.get(key) or []
        raw = values[index] if index < len(values) else None
        out = _number(raw)
        return default if out is None else out

    precip = value("precipitation_probability")
    wind = value("windspeed_10m")
    temp = value("temperature_2m", 20.0)
    code = value("weathercode")
    severity = (
        0.45 * float(np.clip((precip - 40.0) / 60.0, 0.0, 1.0))
        + 0.30 * float(np.clip((wind - 20.0) / 30.0, 0.0, 1.0))
        + 0.15 * float(np.clip((abs(temp - 20.0) - 8.0) / 18.0, 0.0, 1.0))
        + (0.10 if code >= 95 else 0.0)
    )
    return float(np.clip(severity, 0.0, 1.0)), temp


def _matchday_base_row(
    *,
    match_id: str,
    kickoff: pd.Timestamp,
    home_team: str,
    away_team: str,
    competition: str,
    source: str,
    available_at: str,
    home_team_id: str = "",
    away_team_id: str = "",
    venue_name: str = "",
    venue_city: str = "",
    venue_country: str = "",
    venue_lat: float | None = None,
    venue_lon: float | None = None,
) -> dict[str, Any]:
    return {
        "match_id": match_id,
        "kickoff_utc": kickoff.isoformat(),
        "home_team": home_team,
        "away_team": away_team,
        "competition": competition,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "venue_name": venue_name,
        "venue_city": venue_city,
        "venue_country": venue_country,
        "venue_lat": venue_lat,
        "venue_lon": venue_lon,
        "source_available_at_utc": available_at,
        "pit_verified": True,
        "starter_status": "EXPECTED",
        "matchday_available_at_utc": available_at,
        "matchday_pit_verified": True,
        "matchday_source": source,
        "matchday_signal_confidence": 0.30,
        "matchday_injury_impact_home": 0.0,
        "matchday_injury_impact_away": 0.0,
        "matchday_lineup_impact_home": 0.0,
        "matchday_lineup_impact_away": 0.0,
        "matchday_weather_penalty_home": 0.0,
        "matchday_weather_penalty_away": 0.0,
        "matchday_rest_diff_hours": np.nan,
        "matchday_market_p_home": np.nan,
        "matchday_market_p_draw": np.nan,
        "matchday_market_p_away": np.nan,
        "matchday_odds_home": np.nan,
        "matchday_odds_draw": np.nan,
        "matchday_odds_away": np.nan,
        "matchday_market_provider": "",
        "matchday_weather_severity": np.nan,
    }


def _sofascore_competition(event: dict[str, Any]) -> str | None:
    tournament = event.get("tournament") or event.get("uniqueTournament") or {}
    name = str(tournament.get("name") or "").strip()
    return SOFASCORE_COMPETITIONS.get(name)


def parse_sofascore_event(event: dict[str, Any]) -> dict[str, Any] | None:
    competition = _sofascore_competition(event)
    if competition is None:
        return None
    event_id = event.get("id")
    try:
        kickoff = pd.Timestamp(datetime.fromtimestamp(int(event["startTimestamp"]), tz=timezone.utc))
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    home = event.get("homeTeam") or {}
    away = event.get("awayTeam") or {}
    home_name = str(home.get("name") or home.get("shortName") or "").strip()
    away_name = str(away.get("name") or away.get("shortName") or "").strip()
    if event_id is None or not home_name or not away_name:
        return None
    venue = event.get("venue") or {}
    country = venue.get("country") or {}
    coords = event.get("venueCoordinates") or venue.get("coordinates") or {}
    canonical = f"sofascore|{event_id}|{competition}|{home_name}|{away_name}"
    match_id = "sofa:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
    return {
        "match_id": match_id,
        "sofascore_event_id": str(event_id),
        "home_team_id": str(home.get("id") or ""),
        "away_team_id": str(away.get("id") or ""),
        "kickoff_utc": kickoff.isoformat(),
        "home_team": home_name,
        "away_team": away_name,
        "competition": competition,
        "venue_name": str(venue.get("name") or "").strip(),
        "venue_city": str(venue.get("city") or "").strip(),
        "venue_country": str(country.get("name") or "").strip(),
        "venue_lat": _number(coords.get("latitude")),
        "venue_lon": _number(coords.get("longitude")),
    }


def parse_football_data_fixtures(
    frame: pd.DataFrame,
    *,
    now: pd.Timestamp,
    horizon_hours: float,
    available_at: str,
    max_events: int,
) -> list[dict[str, Any]]:
    required = {"Div", "Date", "Time", "HomeTeam", "AwayTeam"}
    if not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise RuntimeError(f"football-data fixtures missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    upper = now + pd.Timedelta(hours=float(horizon_hours))
    temp = frame.copy()
    temp["Div"] = temp["Div"].astype("string").str.strip()
    temp = temp[temp["Div"].isin(FOOTBALL_DATA_DIVISIONS)]
    for _, raw in temp.iterrows():
        div = str(raw["Div"]).strip()
        competition = FOOTBALL_DATA_DIVISIONS[div]
        tz_name = FOOTBALL_DATA_TZ.get(competition)
        if tz_name is None:
            continue
        date_text = str(raw.get("Date", "")).strip()
        time_text = str(raw.get("Time", "")).strip()
        if not date_text or not time_text or date_text == "nan" or time_text == "nan":
            continue
        naive = pd.to_datetime(f"{date_text} {time_text}", dayfirst=True, errors="coerce")
        if pd.isna(naive):
            continue
        kickoff = pd.Timestamp(naive).tz_localize(
            ZoneInfo(tz_name), ambiguous="NaT", nonexistent="NaT"
        ).tz_convert("UTC")
        if pd.isna(kickoff) or kickoff <= now or kickoff > upper:
            continue
        home = str(raw.get("HomeTeam", "")).strip()
        away = str(raw.get("AwayTeam", "")).strip()
        if not home or not away or home == "nan" or away == "nan":
            continue
        canonical = f"football-data|{div}|{kickoff.isoformat()}|{home}|{away}"
        match_id = "fdx:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
        row = _matchday_base_row(
            match_id=match_id,
            kickoff=kickoff,
            home_team=home,
            away_team=away,
            competition=competition,
            source="football-data.co.uk",
            available_at=available_at,
        )
        odds_candidates = [
            ("AvgH", "AvgD", "AvgA", "football-data:average"),
            ("B365H", "B365D", "B365A", "football-data:b365"),
            ("PSH", "PSD", "PSA", "football-data:ps"),
        ]
        for hcol, dcol, acol, provider in odds_candidates:
            values = [_number(raw.get(c)) for c in (hcol, dcol, acol)]
            if all(v is not None and v > 1.0 for v in values):
                odds = tuple(float(v) for v in values)
                probs = _devig(odds)
                row["matchday_odds_home"], row["matchday_odds_draw"], row["matchday_odds_away"] = odds
                row["matchday_market_p_home"], row["matchday_market_p_draw"], row["matchday_market_p_away"] = probs
                row["matchday_market_provider"] = provider
                row["matchday_signal_confidence"] = 0.55
                break
        rows.append(row)
        if len(rows) >= int(max_events):
            break
    return rows


def _sofascore_missing_impact(items: list[dict[str, Any]] | None) -> tuple[float, int]:
    if items is None:
        return np.nan, 0
    total = 0.0
    severe = 0
    for item in items or []:
        if not isinstance(item, dict):
            continue
        reason = str(item.get("reason") or item.get("status") or "").strip().lower()
        if "injur" in reason:
            weight = 1.0
        elif "suspend" in reason:
            weight = 0.85
        elif "ill" in reason or "sick" in reason:
            weight = 0.60
        else:
            weight = 0.35
        total += weight
        severe += int(weight >= 0.85)
    return float(np.clip(total / 4.0, 0.0, 1.0)), severe


def _enrich_sofascore_lineup(fetcher: ExternalFetcher, row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    event_id = str(row.get("sofascore_event_id") or "")
    if not event_id:
        return row, []
    payload, at = _get_json(
        fetcher,
        "sofascore_lineups",
        f"https://www.sofascore.com/api/v1/event/{event_id}/lineups",
    )
    retrieval_times = [at]
    confirmed = payload.get("confirmed") is True
    counts: dict[str, int] = {}
    for side in ("home", "away"):
        group = payload.get(side) or {}
        players = group.get("players") or []
        starters = [
            p for p in players
            if isinstance(p, dict) and (
                p.get("starter") is True or p.get("substitute") is False
            )
        ]
        counts[side] = len(starters)
        impact, severe = _sofascore_missing_impact(group.get("missingPlayers"))
        row[f"matchday_injury_impact_{side}"] = impact
        row["matchday_signal_confidence"] = max(
            float(row.get("matchday_signal_confidence", 0.30)),
            float(np.clip(0.45 + severe * 0.10, 0.45, 0.85)),
        )
    if confirmed and counts.get("home", 0) >= 11 and counts.get("away", 0) >= 11:
        row["starter_status"] = "ANNOUNCED"
        row["matchday_lineup_impact_home"] = 0.5
        row["matchday_lineup_impact_away"] = 0.5
        row["matchday_signal_confidence"] = max(
            float(row.get("matchday_signal_confidence", 0.30)), 0.85
        )
    row["matchday_source"] = "sofascore"
    return row, retrieval_times


def _event_core(event: dict[str, Any], competition: str) -> dict[str, Any] | None:
    competitions = event.get("competitions") or []
    if not competitions:
        return None
    comp = competitions[0]
    home = next((x for x in comp.get("competitors", []) or [] if x.get("homeAway") == "home"), None)
    away = next((x for x in comp.get("competitors", []) or [] if x.get("homeAway") == "away"), None)
    kickoff = _ts(event.get("date") or comp.get("startDate"))
    if not home or not away or kickoff is None:
        return None
    venue = comp.get("venue") or {}
    address = venue.get("address") or {}
    return {
        "match_id": f"espn:{event.get('id')}",
        "espn_event_id": str(event.get("id")),
        "espn_league": ESPN_LEAGUES[competition],
        "kickoff_utc": kickoff.isoformat(),
        "home_team": str((home.get("team") or {}).get("displayName") or home.get("id")),
        "away_team": str((away.get("team") or {}).get("displayName") or away.get("id")),
        "home_team_id": str((home.get("team") or {}).get("id") or home.get("id") or ""),
        "away_team_id": str((away.get("team") or {}).get("id") or away.get("id") or ""),
        "competition": competition,
        "venue_name": str(venue.get("fullName") or ""),
        "venue_city": str(address.get("city") or ""),
        "venue_country": str(address.get("country") or ""),
        "venue_lat": _number((venue.get("coordinates") or {}).get("latitude")),
        "venue_lon": _number((venue.get("coordinates") or {}).get("longitude")),
    }


def _weather(
    fetcher: ExternalFetcher,
    row: dict[str, Any],
    kickoff: pd.Timestamp,
) -> tuple[float, str | None, float | None, float | None]:
    lat, lon = row.get("venue_lat"), row.get("venue_lon")
    retrieved = None
    if lat is None or lon is None:
        name = f"{row.get('venue_city', '')}, {row.get('venue_country', '')}".strip(", ")
        if not name:
            return np.nan, None, lat, lon
        geo, retrieved = _get_json(
            fetcher,
            "open_meteo_geocoding",
            "https://geocoding-api.open-meteo.com/v1/search",
            {"name": name, "count": 1, "language": "en", "format": "json"},
        )
        results = geo.get("results") or []
        if not results:
            return np.nan, retrieved, lat, lon
        lat = _number(results[0].get("latitude"))
        lon = _number(results[0].get("longitude"))
    if lat is None or lon is None:
        return np.nan, retrieved, lat, lon
    weather, retrieved = _get_json(
        fetcher,
        "open_meteo_forecast",
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,precipitation_probability,windspeed_10m,weathercode",
            "timezone": "UTC",
            "forecast_days": 2,
        },
    )
    severity, _ = parse_weather_severity(weather, kickoff)
    return severity, retrieved, lat, lon


def _collect_sofascore_day(
    fetcher: ExternalFetcher,
    *,
    day: datetime,
    now_ts: pd.Timestamp,
    horizon_hours: float,
    max_events: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], str | None]:
    date_key = day.strftime("%Y-%m-%d")
    try:
        payload, scheduled_at = _get_json(
            fetcher,
            "sofascore_scheduled_events",
            f"https://www.sofascore.com/api/v1/sport/football/scheduled-events/{date_key}",
        )
    except Exception as exc:
        return [], [{"source": "sofascore_scheduled_events", "error": f"{type(exc).__name__}: {exc}"}], None
    errors: list[dict[str, str]] = []
    parsed: list[dict[str, Any]] = []
    upper = now_ts + pd.Timedelta(hours=float(horizon_hours))
    for event in payload.get("events", []) or []:
        core = parse_sofascore_event(event)
        if not core:
            continue
        kickoff = _ts(core["kickoff_utc"])
        if kickoff is None or kickoff <= now_ts or kickoff > upper:
            continue
        row = _matchday_base_row(
            match_id=core["match_id"],
            kickoff=kickoff,
            home_team=core["home_team"],
            away_team=core["away_team"],
            competition=core["competition"],
            source="sofascore",
            available_at=scheduled_at,
            home_team_id=core.get("home_team_id", ""),
            away_team_id=core.get("away_team_id", ""),
            venue_name=core.get("venue_name", ""),
            venue_city=core.get("venue_city", ""),
            venue_country=core.get("venue_country", ""),
            venue_lat=core.get("venue_lat"),
            venue_lon=core.get("venue_lon"),
        )
        row["sofascore_event_id"] = core["sofascore_event_id"]
        parsed.append(row)
    parsed.sort(key=lambda x: (str(x["kickoff_utc"]), str(x["match_id"])))
    for row in parsed[:min(int(max_events), SOFASCORE_LINEUP_ENRICH_LIMIT)]:
        kickoff = _ts(row["kickoff_utc"])
        if kickoff is None:
            continue
        times = [scheduled_at]
        try:
            row, extra = _enrich_sofascore_lineup(fetcher, row)
            times.extend(extra)
        except Exception as exc:
            errors.append({
                "source": "sofascore_lineups",
                "match_id": str(row["match_id"]),
                "error": f"{type(exc).__name__}: {exc}",
            })
        try:
            severity, at, lat, lon = _weather(fetcher, row, kickoff)
            if at:
                times.append(at)
            row["venue_lat"], row["venue_lon"] = lat, lon
            row["matchday_weather_severity"] = severity
            row["matchday_weather_penalty_home"] = severity
            row["matchday_weather_penalty_away"] = severity
        except Exception as exc:
            errors.append({
                "source": "open_meteo",
                "match_id": str(row["match_id"]),
                "error": f"{type(exc).__name__}: {exc}",
            })
        row["matchday_available_at_utc"] = max(times)
        row["matchday_pit_verified"] = all(pd.Timestamp(x).tzinfo is not None for x in times)
    return parsed[:int(max_events)], errors, scheduled_at


def _collect_football_data_fallback(
    fetcher: ExternalFetcher,
    *,
    now_ts: pd.Timestamp,
    horizon_hours: float,
    max_events: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], str | None]:
    try:
        frame, retrieved_at = _get_csv(
            fetcher, "football_data_fixtures", FOOTBALL_DATA_FIXTURES_URL
        )
        rows = parse_football_data_fixtures(
            frame,
            now=now_ts,
            horizon_hours=horizon_hours,
            available_at=retrieved_at,
            max_events=max_events,
        )
        return rows, [], retrieved_at
    except Exception as exc:
        return [], [{
            "source": "football_data_fixtures",
            "error": f"{type(exc).__name__}: {exc}",
        }], None


def collect_matchday_snapshots(
    *,
    days: int = 2,
    horizon_hours: float = DETAILED_HORIZON_HOURS,
    max_events: int = MAX_EVENTS,
    cache_dir: str = "cache/external",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collect current evidence; no historical PIT claim is made."""
    now = _now()
    now_ts = pd.Timestamp(now)
    fetcher = ExternalFetcher(cache_dir=cache_dir, timeout=20.0, retries=3, backoff=1.0)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen: set[str] = set()

    fallback_usage: list[dict[str, Any]] = []
    for day_offset in range(max(1, int(days))):
        day_start = now + pd.Timedelta(days=day_offset)
        date = day_start.strftime("%Y%m%d")
        before_day_rows = len(rows)
        for competition, league in ESPN_LEAGUES.items():
            try:
                scoreboard, scoreboard_at = _get_json(
                    fetcher,
                    "espn_scoreboard",
                    f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard",
                    {"dates": date},
                )
            except Exception as exc:
                errors.append({
                    "source": "espn_scoreboard",
                    "competition": competition,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue

            for event in scoreboard.get("events", []) or []:
                core = _event_core(event, competition)
                if not core or core["match_id"] in seen:
                    continue
                kickoff = _ts(core["kickoff_utc"])
                if kickoff is None or kickoff <= now_ts:
                    continue
                seen.add(core["match_id"])

                row: dict[str, Any] = {
                    **core,
                    "source_available_at_utc": scoreboard_at,
                    "pit_verified": True,
                    "starter_status": "EXPECTED",
                    "matchday_available_at_utc": scoreboard_at,
                    "matchday_pit_verified": True,
                    "matchday_source": "espn_scoreboard",
                    "matchday_signal_confidence": 0.40,
                    "matchday_injury_impact_home": np.nan,
                    "matchday_injury_impact_away": np.nan,
                    "matchday_lineup_impact_home": np.nan,
                    "matchday_lineup_impact_away": np.nan,
                    "matchday_weather_penalty_home": np.nan,
                    "matchday_weather_penalty_away": np.nan,
                    "matchday_rest_diff_hours": np.nan,
                    "matchday_market_p_home": np.nan,
                    "matchday_market_p_draw": np.nan,
                    "matchday_market_p_away": np.nan,
                    "matchday_odds_home": np.nan,
                    "matchday_odds_draw": np.nan,
                    "matchday_odds_away": np.nan,
                    "matchday_market_provider": "",
                    "matchday_weather_severity": np.nan,
                }

                near = (kickoff - now_ts).total_seconds() / 3600.0 <= float(horizon_hours)
                if near:
                    retrieval_times = [scoreboard_at]
                    try:
                        summary, at = _get_json(
                            fetcher,
                            "espn_summary",
                            f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/summary",
                            {"event": core["espn_event_id"]},
                        )
                        retrieval_times.append(at)
                        odds, market, provider = parse_market_odds(summary)
                        if odds is not None:
                            row["matchday_odds_home"], row["matchday_odds_draw"], row["matchday_odds_away"] = odds
                        if market is not None:
                            row["matchday_market_p_home"], row["matchday_market_p_draw"], row["matchday_market_p_away"] = market
                        row["matchday_market_provider"] = provider
                    except Exception as exc:
                        errors.append({
                            "source": "espn_summary",
                            "match_id": core["match_id"],
                            "error": f"{type(exc).__name__}: {exc}",
                        })

                    injury_ok = 0
                    for side, key in (("home", "home_team_id"), ("away", "away_team_id")):
                        team_id = row[key]
                        if not team_id:
                            continue
                        try:
                            payload, at = _get_json(
                                fetcher,
                                "espn_injuries",
                                f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/teams/{team_id}/injuries",
                            )
                            retrieval_times.append(at)
                            impact, _, _ = parse_injury_impact(payload)
                            row[f"matchday_injury_impact_{side}"] = impact
                            injury_ok += 1
                        except Exception as exc:
                            errors.append({
                                "source": "espn_injuries",
                                "match_id": core["match_id"],
                                "error": f"{type(exc).__name__}: {exc}",
                            })

                    starters: dict[str, int] = {}
                    for side, key in (("home", "home_team_id"), ("away", "away_team_id")):
                        team_id = row[key]
                        if not team_id:
                            continue
                        url = (
                            f"https://sports.core.api.espn.com/v2/sports/soccer/leagues/{league}"
                            f"/events/{core['espn_event_id']}/competitions/{core['espn_event_id']}"
                            f"/competitors/{team_id}/roster"
                        )
                        try:
                            payload, at = _get_json(fetcher, "espn_event_roster", url)
                            retrieval_times.append(at)
                            count, _ = parse_event_roster(payload)
                            starters[side] = count
                        except Exception as exc:
                            errors.append({
                                "source": "espn_event_roster",
                                "match_id": core["match_id"],
                                "error": f"{type(exc).__name__}: {exc}",
                            })
                    if starters.get("home", 0) >= 11 and starters.get("away", 0) >= 11:
                        row["starter_status"] = "ANNOUNCED"
                        row["matchday_signal_confidence"] = max(row["matchday_signal_confidence"], 0.85)

                    rest_values: dict[str, float] = {}
                    for side, key in (("home", "home_team_id"), ("away", "away_team_id")):
                        team_id = row[key]
                        if not team_id:
                            continue
                        try:
                            schedule, at = _get_json(
                                fetcher,
                                "espn_team_schedule",
                                f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/teams/{team_id}/schedule",
                                {"limit": 30},
                            )
                            retrieval_times.append(at)
                            prior = [
                                _ts(e.get("date") or e.get("startDate"))
                                for e in schedule.get("events", []) or []
                            ]
                            prior = [x for x in prior if x is not None and x < kickoff]
                            if prior:
                                rest_values[side] = float((kickoff - max(prior)).total_seconds() / 3600.0)
                        except Exception as exc:
                            errors.append({
                                "source": "espn_team_schedule",
                                "match_id": core["match_id"],
                                "error": f"{type(exc).__name__}: {exc}",
                            })
                    if "home" in rest_values and "away" in rest_values:
                        row["matchday_rest_diff_hours"] = rest_values["home"] - rest_values["away"]

                    try:
                        severity, at, lat, lon = _weather(fetcher, row, kickoff)
                        if at:
                            retrieval_times.append(at)
                        row["venue_lat"], row["venue_lon"] = lat, lon
                        row["matchday_weather_severity"] = severity
                        # Same weather is not turned into an invented home/away edge.
                        row["matchday_weather_penalty_home"] = severity
                        row["matchday_weather_penalty_away"] = severity
                    except Exception as exc:
                        errors.append({
                            "source": "open_meteo",
                            "match_id": core["match_id"],
                            "error": f"{type(exc).__name__}: {exc}",
                        })

                    row["matchday_available_at_utc"] = max(retrieval_times)
                    row["matchday_pit_verified"] = all(
                        pd.Timestamp(x).tzinfo is not None and pd.Timestamp(x) <= now_ts
                        for x in retrieval_times
                    )
                    quality = [
                        float(bool(row["matchday_market_provider"])),
                        float(injury_ok == 2),
                        float(row["starter_status"] == "ANNOUNCED"),
                        float(pd.notna(row["matchday_weather_severity"])),
                        float(pd.notna(row["matchday_rest_diff_hours"])),
                    ]
                    row["matchday_signal_confidence"] = float(
                        np.clip(max(row["matchday_signal_confidence"], np.mean(quality)), 0.0, 1.0)
                    )
                    row["matchday_source"] = "espn+open_meteo"

                rows.append(row)
                if len(rows) >= max_events:
                    break
            if len(rows) >= max_events:
                break
        if len(rows) >= max_events:
            break

        if len(rows) == before_day_rows and len(rows) < max_events:
            day_rows, day_errors, _ = _collect_sofascore_day(
                fetcher,
                day=day_start,
                now_ts=now_ts,
                horizon_hours=float(horizon_hours),
                max_events=max_events - len(rows),
            )
            if day_rows:
                rows.extend(day_rows)
                fallback_usage.append({"provider": "sofascore", "date": date, "rows": len(day_rows)})
            errors.extend(day_errors)
            if not day_rows and len(rows) < max_events:
                fd_rows, fd_errors, _ = _collect_football_data_fallback(
                    fetcher,
                    now_ts=now_ts,
                    horizon_hours=float(horizon_hours),
                    max_events=max_events - len(rows),
                )
                day_date = day_start.date()
                fd_rows = [
                    row for row in fd_rows
                    if _ts(row["kickoff_utc"]) is not None and _ts(row["kickoff_utc"]).date() == day_date
                ]
                if fd_rows:
                    rows.extend(fd_rows)
                    fallback_usage.append({"provider": "football-data.co.uk", "date": date, "rows": len(fd_rows)})
                errors.extend(fd_errors)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["kickoff_utc"] = pd.to_datetime(frame["kickoff_utc"], utc=True)
        frame["source_available_at_utc"] = pd.to_datetime(frame["source_available_at_utc"], utc=True)
        frame["matchday_available_at_utc"] = pd.to_datetime(frame["matchday_available_at_utc"], utc=True)
        frame = (
            frame.drop_duplicates("match_id")
            .sort_values(["kickoff_utc", "match_id"], kind="mergesort")
            .reset_index(drop=True)
        )

    snapshot_finished = _now()
    if not frame.empty:
        available = pd.to_datetime(frame["matchday_available_at_utc"], utc=True, errors="coerce")
        frame["matchday_pit_verified"] = available.notna() & (available <= pd.Timestamp(snapshot_finished))
    status = {
        "status": "COLLECTED" if not frame.empty else (
            "DEFERRED_EXTERNAL_SOURCE" if errors else "NO_UPCOMING_FIXTURES"
        ),
        "prediction_time_utc": iso_utc(snapshot_finished),
        "snapshot_started_at_utc": iso_utc(now),
        "snapshot_finished_at_utc": iso_utc(snapshot_finished),
        "rows": int(len(frame)),
        "errors": errors,
        "fallback_usage": fallback_usage,
        "historical_pit_claim": False,
        "current_snapshot_pit_basis": "source_retrieval_time_and_snapshot_finish",
        "free_keyless_default": True,
    }
    return frame, status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/future_matchday_fixtures.csv")
    parser.add_argument("--status", default="artifacts/matchday_intelligence_status.json")
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--horizon-hours", type=float, default=DETAILED_HORIZON_HOURS)
    parser.add_argument("--max-events", type=int, default=MAX_EVENTS)
    parser.add_argument("--cache-dir", default="cache/external")
    args = parser.parse_args()

    output = Path(args.output)
    status_path = Path(args.status)
    output.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        frame, status = collect_matchday_snapshots(
            days=max(1, args.days),
            horizon_hours=max(1.0, args.horizon_hours),
            max_events=max(1, args.max_events),
            cache_dir=args.cache_dir,
        )
    except Exception as exc:
        status = {
            "status": "FAILED_INTERNAL",
            "error": f"{type(exc).__name__}: {exc}",
            "free_keyless_default": True,
        }
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
        return 1

    if frame.empty:
        pd.DataFrame(columns=[
            "match_id", "kickoff_utc", "home_team", "away_team", "competition",
            "source_available_at_utc", "pit_verified", "starter_status",
            "matchday_available_at_utc", "matchday_pit_verified",
        ]).to_csv(output, index=False)
    else:
        frame.to_csv(output, index=False)
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
