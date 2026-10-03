from __future__ import annotations

"""Rich, current-matchday enrichment built on the existing acquisition layer.

This module deliberately remains research-only evidence. It reuses the existing
PIT-aware fetch/cache layer and existing ESPN/SofaScore helpers, but never turns
current retrieval time into historical publication-time proof.

The output is intentionally broader than the production feature surface:
- match/event metadata
- competition/season/round/status
- referee/manager context
- confirmed/expected lineup structure
- multi-provider market statistics
- H2H and recent-form summaries
- schedule congestion/rest
- detailed weather
- raw provider payload lineage in JSONL
"""

import argparse
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.external_fetch import ExternalFetcher, iso_utc
from src.data.matchday_intelligence_fetch import ESPN_LEAGUES, _get_json, _number, _ts

DEFAULT_DETAIL_HOURS = 48.0
DEFAULT_MAX_MATCHES = 48
DEFAULT_WORKERS = 8
MAX_H2H_EVENTS = 10
MAX_RECENT_EVENTS = 10


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def _safe_int(value: Any) -> int | None:
    number = _number(value)
    if number is None or not math.isfinite(number):
        return None
    return int(number)


def _round(value: Any, digits: int = 6) -> float | None:
    number = _number(value)
    return None if number is None else round(float(number), digits)


def _event_start_timestamp(value: Any) -> pd.Timestamp | None:
    """Parse provider event timestamps without mistaking Unix seconds for nanoseconds."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if math.isfinite(number) and abs(number) >= 1_000_000_000:
            try:
                return pd.Timestamp.fromtimestamp(number, tz="UTC")
            except (OverflowError, OSError, ValueError):
                return None
    return _ts(value)


def _json_hash(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _hours_to_kickoff(kickoff: Any, now: pd.Timestamp) -> float | None:
    stamp = _ts(kickoff)
    if stamp is None:
        return None
    return float((stamp - now).total_seconds() / 3600.0)


def _competitor_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    competitions = summary.get("competitions") or []
    if not competitions or not isinstance(competitions[0], dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for item in competitions[0].get("competitors", []) or []:
        if not isinstance(item, dict):
            continue
        side = _safe_str(item.get("homeAway")).lower()
        if side in {"home", "away"}:
            out[side] = item
    return out


def _record_summary(competitor: dict[str, Any]) -> str:
    records = competitor.get("records") or []
    summaries = [
        _safe_str(item.get("summary"))
        for item in records
        if isinstance(item, dict) and _safe_str(item.get("summary"))
    ]
    return " | ".join(dict.fromkeys(summaries))


def _parse_espn_summary(payload: dict[str, Any]) -> dict[str, Any]:
    competitions = payload.get("competitions") or []
    competition = competitions[0] if competitions and isinstance(competitions[0], dict) else {}
    status = competition.get("status") or {}
    status_type = status.get("type") or {}
    season = competition.get("season") or {}
    week = competition.get("week") or {}
    competitors = _competitor_map(payload)

    def rank(side: str) -> int | None:
        item = competitors.get(side, {})
        curated = item.get("curatedRank") or {}
        return _safe_int(curated.get("current"))

    def form(side: str) -> str:
        return _safe_str(competitors.get(side, {}).get("form"))

    officials = competition.get("officials") or []
    referee_names: list[str] = []
    for item in officials:
        if not isinstance(item, dict):
            continue
        name = _safe_str(
            item.get("fullName")
            or item.get("displayName")
            or (item.get("athlete") or {}).get("displayName")
        )
        if name:
            referee_names.append(name)

    broadcasts = []
    for item in competition.get("broadcasts", []) or []:
        if not isinstance(item, dict):
            continue
        names = [
            _safe_str(item.get("names")),
            _safe_str(item.get("market")),
            _safe_str((item.get("media") or {}).get("shortName")),
        ]
        value = " / ".join(x for x in names if x)
        if value:
            broadcasts.append(value)

    news = payload.get("news") or []
    latest_news = []
    for item in news:
        if not isinstance(item, dict):
            continue
        stamp = _ts(item.get("published") or item.get("publishedAt"))
        if stamp is not None:
            latest_news.append(stamp)

    return {
        "rich_espn_status": _safe_str(
            status_type.get("description")
            or status_type.get("shortDetail")
            or status_type.get("name")
        ),
        "rich_espn_status_state": _safe_str(status_type.get("state")),
        "rich_espn_season": _safe_str(season.get("displayName")),
        "rich_espn_season_year": _safe_int(season.get("year")),
        "rich_espn_week": _safe_int(week.get("number")),
        "rich_espn_week_text": _safe_str(week.get("text")),
        "rich_espn_home_rank": rank("home"),
        "rich_espn_away_rank": rank("away"),
        "rich_espn_home_form": form("home"),
        "rich_espn_away_form": form("away"),
        "rich_espn_home_record": _record_summary(competitors.get("home", {})),
        "rich_espn_away_record": _record_summary(competitors.get("away", {})),
        "rich_espn_neutral_site": bool(
            competition.get("neutralSite") or competition.get("neutral")
        ),
        "rich_referees": " | ".join(dict.fromkeys(referee_names)),
        "rich_broadcasts": " | ".join(dict.fromkeys(broadcasts)),
        "rich_news_count": int(len(news)),
        "rich_latest_news_published_at_utc": (
            max(latest_news).isoformat() if latest_news else ""
        ),
    }


def _market_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in payload.get("odds", []) or []:
        if not isinstance(item, dict):
            continue
        home = item.get("homeTeamOdds") or {}
        draw = item.get("drawOdds") or item.get("drawTeamOdds") or {}
        away = item.get("awayTeamOdds") or {}
        values = [
            _number(home.get("decimalValue")) or _number(home.get("value")),
            _number(draw.get("decimalValue")) or _number(draw.get("value")),
            _number(away.get("decimalValue")) or _number(away.get("value")),
        ]
        if any(v is None or v <= 1.0 for v in values):
            continue
        odds = np.asarray(values, dtype=float)
        inv = 1.0 / odds
        overround = float(inv.sum())
        if not math.isfinite(overround) or overround <= 0:
            continue
        probs = inv / overround
        provider = _safe_str((item.get("provider") or {}).get("name")) or "UNKNOWN"
        priority = _safe_int((item.get("provider") or {}).get("priority"))
        last_updated = _safe_str(
            item.get("lastUpdated")
            or item.get("lastUpdate")
            or item.get("updateTimestamp")
            or item.get("asOf")
        )
        spread = _number(item.get("spread") or item.get("pointSpread"))
        total = _number(item.get("overUnder") or item.get("total"))
        if total is None:
            total = _number((item.get("total") or {}).get("value")) if isinstance(item.get("total"), dict) else None
        rows.append(
            {
                "provider": provider,
                "provider_priority": priority,
                "last_updated": last_updated,
                "spread": spread,
                "total": total,
                "home_odds": float(odds[0]),
                "draw_odds": float(odds[1]),
                "away_odds": float(odds[2]),
                "home_prob": float(probs[0]),
                "draw_prob": float(probs[1]),
                "away_prob": float(probs[2]),
                "overround": overround,
            }
        )
    return rows


def _market_summary(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        if isinstance(payload, dict):
            rows.extend(_market_rows(payload))
    if not rows:
        return {"rich_market_provider_count": np.nan, "rich_market_provider_names": ""}
    frame = pd.DataFrame(rows).drop_duplicates(
        subset=["provider", "home_odds", "draw_odds", "away_odds"],
        keep="first",
    )
    out: dict[str, Any] = {
        "rich_market_provider_count": int(frame["provider"].nunique()),
        "rich_market_provider_names": " | ".join(
            sorted(frame["provider"].astype(str).unique())
        ),
    }
    for field in (
        "provider_priority",
        "spread",
        "total",
        "home_odds",
        "draw_odds",
        "away_odds",
        "home_prob",
        "draw_prob",
        "away_prob",
        "overround",
    ):
        series = pd.to_numeric(frame[field], errors="coerce").dropna()
        if series.empty:
            out[f"rich_market_{field}_mean"] = np.nan
            out[f"rich_market_{field}_std"] = np.nan
            out[f"rich_market_{field}_min"] = np.nan
            out[f"rich_market_{field}_max"] = np.nan
        else:
            out[f"rich_market_{field}_mean"] = float(series.mean())
            out[f"rich_market_{field}_std"] = float(series.std(ddof=0))
            out[f"rich_market_{field}_min"] = float(series.min())
            out[f"rich_market_{field}_max"] = float(series.max())
    timestamps = pd.to_datetime(frame["last_updated"], utc=True, errors="coerce").dropna()
    out["rich_market_latest_observation_at_utc"] = (
        timestamps.max().isoformat() if not timestamps.empty else ""
    )
    out["rich_market_observation_timestamp_count"] = int(len(timestamps))
    return out


def _parse_weather_payload(payload: dict[str, Any], kickoff: pd.Timestamp) -> dict[str, Any]:
    hourly = payload.get("hourly") or {}
    times = pd.to_datetime(
        pd.Series(hourly.get("time") or []),
        utc=True,
        errors="coerce",
    )
    if times.empty or times.isna().all():
        return {}
    deltas = (times - kickoff).dt.total_seconds().abs().to_numpy(dtype=float)
    if deltas.size == 0 or not np.isfinite(deltas).any():
        return {}
    index = int(np.nanargmin(deltas))

    variable_names = (
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "apparent_temperature",
        "precipitation_probability",
        "precipitation",
        "rain",
        "showers",
        "snowfall",
        "visibility",
        "pressure_msl",
        "surface_pressure",
        "cloud_cover",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
        "is_day",
        "sunshine_duration",
        "cape",
        "weather_code",
    )
    out: dict[str, Any] = {}
    for name in variable_names:
        values = hourly.get(name)
        if not isinstance(values, list) or index >= len(values):
            continue
        value = values[index]
        if value is None:
            continue
        key = f"rich_weather_{name}"
        if name in {"is_day", "weather_code"}:
            out[key] = _safe_int(value)
        else:
            out[key] = _round(value, 4)
    out["rich_weather_nearest_valid_time_utc"] = times.iloc[index].isoformat()
    return out


def _parse_sofa_event(payload: dict[str, Any]) -> dict[str, Any]:
    tournament = payload.get("tournament") or {}
    unique = tournament.get("uniqueTournament") or {}
    season = payload.get("season") or {}
    round_info = payload.get("roundInfo") or {}
    status = payload.get("status") or {}
    referee_names = []
    for item in payload.get("referees", []) or []:
        if not isinstance(item, dict):
            continue
        name = _safe_str((item.get("referee") or {}).get("name") or item.get("name"))
        if name:
            referee_names.append(name)
    venue = payload.get("venue") or {}
    managers = payload.get("managers") or {}
    return {
        "rich_sofa_tournament": _safe_str(unique.get("name") or tournament.get("name")),
        "rich_sofa_season": _safe_str(season.get("name")),
        "rich_sofa_round": _safe_int(round_info.get("round")),
        "rich_sofa_round_name": _safe_str(
            round_info.get("name") or round_info.get("slug")
        ),
        "rich_sofa_status_code": _safe_int(status.get("code")),
        "rich_sofa_status_type": _safe_str(status.get("type")),
        "rich_sofa_referees": " | ".join(dict.fromkeys(referee_names)),
        "rich_sofa_venue": _safe_str(venue.get("name")),
        "rich_sofa_neutral_ground": bool(
            payload.get("neutralGround") or payload.get("isNeutral")
        ),
        "rich_sofa_home_manager": _safe_str((managers.get("home") or {}).get("name")),
        "rich_sofa_away_manager": _safe_str((managers.get("away") or {}).get("name")),
    }


def _parse_lineups(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for side in ("home", "away"):
        group = payload.get(side) or {}
        players = group.get("players") or []
        starters: list[str] = []
        starter_names: list[str] = []
        substitutes: list[str] = []
        starter_positions: list[str] = []
        starter_values: list[float] = []
        starter_ages: list[float] = []
        starter_heights: list[float] = []
        starter_rating_values: list[float] = []
        position_counts = {"G": 0, "D": 0, "M": 0, "F": 0, "UNK": 0}
        foot_counts = {"left": 0, "right": 0, "both": 0, "unknown": 0}
        captain = ""

        def scalar_number(value: Any, *keys: str) -> float | None:
            number = _number(value)
            if number is not None:
                return number
            if isinstance(value, dict):
                for key in keys:
                    number = _number(value.get(key))
                    if number is not None:
                        return number
            return None

        for player in players:
            if not isinstance(player, dict):
                continue
            athlete = player.get("player") or {}
            player_id = athlete.get("id") or player.get("playerId")
            token = _safe_str(player_id)
            name = _safe_str(
                athlete.get("name")
                or athlete.get("shortName")
                or athlete.get("displayName")
                or athlete.get("slug")
            )
            if player.get("captain") is True:
                captain = token
            starter = player.get("starter") is True or player.get("substitute") is False
            if starter:
                if token:
                    starters.append(token)
                if name:
                    starter_names.append(name)
                position_obj = athlete.get("position") or player.get("position") or ""
                position = _safe_str(
                    (position_obj.get("shortName") if isinstance(position_obj, dict) else None)
                    or (position_obj.get("name") if isinstance(position_obj, dict) else None)
                    or (position_obj.get("slug") if isinstance(position_obj, dict) else None)
                    or position_obj
                ).upper()
                if position:
                    starter_positions.append(position)
                bucket = "UNK"
                if position.startswith("G") or "KEEPER" in position:
                    bucket = "G"
                elif position.startswith("D") or "DEF" in position:
                    bucket = "D"
                elif position.startswith("M") or "MID" in position:
                    bucket = "M"
                elif position.startswith("F") or "FORWARD" in position or "ATTACK" in position:
                    bucket = "F"
                position_counts[bucket] += 1
                age = scalar_number(athlete.get("age"), "value")
                height = scalar_number(athlete.get("height"), "value", "cm")
                rating = scalar_number(
                    player.get("rating") or athlete.get("rating"), "value"
                )
                market_value = scalar_number(
                    athlete.get("marketValue")
                    or athlete.get("proposedMarketValue")
                    or athlete.get("value"),
                    "value",
                    "amount",
                )
                if age is not None and 12 <= age <= 60:
                    starter_ages.append(age)
                if height is not None and 130 <= height <= 230:
                    starter_heights.append(height)
                if rating is not None and 0 <= rating <= 10:
                    starter_rating_values.append(rating)
                if market_value is not None and market_value >= 0:
                    starter_values.append(market_value)
                foot = _safe_str(
                    athlete.get("preferredFoot")
                    or athlete.get("preferredFootSide")
                    or (athlete.get("foot") or {}).get("name")
                    if isinstance(athlete.get("foot"), dict)
                    else athlete.get("preferredFoot") or athlete.get("preferredFootSide")
                ).lower()
                if foot in foot_counts:
                    foot_counts[foot] += 1
                elif foot == "both feet" or foot == "both":
                    foot_counts["both"] += 1
                else:
                    foot_counts["unknown"] += 1
            elif token:
                substitutes.append(token)

        missing = group.get("missingPlayers")
        missing_count = len(missing) if isinstance(missing, list) else None
        missing_ids: list[str] = []
        missing_names: list[str] = []
        reasons = []
        if isinstance(missing, list):
            for item in missing:
                if not isinstance(item, dict):
                    continue
                missing_player = item.get("player") or item.get("athlete") or {}
                missing_id = _safe_str(
                    item.get("playerId") or missing_player.get("id")
                )
                missing_name = _safe_str(
                    item.get("name")
                    or missing_player.get("name")
                    or missing_player.get("shortName")
                )
                if missing_id:
                    missing_ids.append(missing_id)
                if missing_name:
                    missing_names.append(missing_name)
                reason = _safe_str(item.get("reason") or item.get("status"))
                if reason:
                    reasons.append(reason)
        formation = group.get("formation")
        if isinstance(formation, dict):
            formation = formation.get("formation") or formation.get("name")
        prefix = f"rich_sofa_{side}_"
        out[f"{prefix}formation"] = _safe_str(formation)
        out[f"{prefix}starter_count"] = len(starters) if players else np.nan
        out[f"{prefix}substitute_count"] = len(substitutes) if players else np.nan
        out[f"{prefix}starter_ids"] = "|".join(starters)
        out[f"{prefix}starter_names"] = "|".join(starter_names)
        out[f"{prefix}starter_positions"] = "|".join(starter_positions)
        out[f"{prefix}captain_id"] = captain
        out[f"{prefix}missing_count"] = missing_count
        out[f"{prefix}missing_ids"] = "|".join(dict.fromkeys(missing_ids))
        out[f"{prefix}missing_names"] = "|".join(dict.fromkeys(missing_names))
        out[f"{prefix}missing_reasons"] = " | ".join(dict.fromkeys(reasons))
        out[f"{prefix}starter_avg_age"] = float(np.mean(starter_ages)) if starter_ages else np.nan
        out[f"{prefix}starter_avg_height_cm"] = float(np.mean(starter_heights)) if starter_heights else np.nan
        out[f"{prefix}starter_avg_rating"] = float(np.mean(starter_rating_values)) if starter_rating_values else np.nan
        out[f"{prefix}starter_market_value_sum"] = float(np.sum(starter_values)) if starter_values else np.nan
        for pos in ("G", "D", "M", "F", "UNK"):
            out[f"{prefix}starter_position_count_{pos.lower()}"] = int(position_counts[pos]) if players else np.nan
        for foot in ("left", "right", "both", "unknown"):
            out[f"{prefix}starter_preferred_foot_{foot}_count"] = int(foot_counts[foot]) if players else np.nan
    confirmed = payload.get("confirmed")
    if isinstance(confirmed, bool):
        out["rich_sofa_lineup_confirmed"] = confirmed
    return out


def _event_result(
    event: dict[str, Any],
    team_id: str | None = None,
) -> tuple[str, int, int, str, str] | None:
    if not isinstance(event, dict):
        return None
    status = event.get("status") or {}
    status_type = _safe_str(status.get("type")).lower()
    if status_type and status_type not in {"finished", "ended", "postponed_completed"}:
        return None
    home = event.get("homeTeam") or {}
    away = event.get("awayTeam") or {}
    home_id = _safe_str(home.get("id"))
    away_id = _safe_str(away.get("id"))
    home_score = _safe_int((event.get("homeScore") or {}).get("current"))
    away_score = _safe_int((event.get("awayScore") or {}).get("current"))
    if home_score is None or away_score is None:
        return None
    team = _safe_str(team_id)
    if team and team not in {home_id, away_id}:
        return None
    tournament = event.get("tournament") or {}
    competition = _safe_str(
        (tournament.get("uniqueTournament") or {}).get("name")
        or tournament.get("name")
    )
    if team:
        gf = home_score if home_id == team else away_score
        ga = away_score if home_id == team else home_score
        result = "W" if gf > ga else "D" if gf == ga else "L"
        side = "home" if home_id == team else "away"
    else:
        gf, ga = home_score, away_score
        result = "H" if gf > ga else "D" if gf == ga else "A"
        side = ""
    return result, int(gf), int(ga), side, competition


def _summarize_recent_events(
    events: list[dict[str, Any]],
    team_id: str,
    kickoff: pd.Timestamp,
) -> dict[str, Any]:
    usable = []
    for event in events:
        stamp = _event_start_timestamp(event.get("startTimestamp"))
        if stamp is None or stamp >= kickoff:
            continue
        result = _event_result(event, team_id)
        if result is None:
            continue
        result_token, gf, ga, side, competition = result
        usable.append((stamp, result_token, gf, ga, side, competition))
    usable.sort(key=lambda x: x[0], reverse=True)
    usable = usable[:MAX_RECENT_EVENTS]
    if not usable:
        return {
            "matches": np.nan,
            "wins": np.nan,
            "draws": np.nan,
            "losses": np.nan,
            "gf": np.nan,
            "ga": np.nan,
            "points": np.nan,
            "avg_gf": np.nan,
            "avg_ga": np.nan,
            "gd": np.nan,
            "over25_rate": np.nan,
            "btts_rate": np.nan,
            "last5": "",
            "last_opponent": "",
            "last_competition": "",
            "last_side": "",
            "rest_hours": np.nan,
            "matches_3d": np.nan,
            "matches_7d": np.nan,
            "matches_14d": np.nan,
            "matches_30d": np.nan,
            "competitions_14d": np.nan,
            "away_matches_14d": np.nan,
            "home_matches_14d": np.nan,
        }

    frame = pd.DataFrame(
        usable,
        columns=["stamp", "result", "gf", "ga", "side", "competition"],
    )
    wins = int((frame.result == "W").sum())
    draws = int((frame.result == "D").sum())
    losses = int((frame.result == "L").sum())
    gf = int(frame.gf.sum())
    ga = int(frame.ga.sum())
    points = wins * 3 + draws
    total = (frame.gf + frame.ga).to_numpy(dtype=float)
    btts = ((frame.gf > 0) & (frame.ga > 0)).mean()
    over25 = (total >= 3).mean()
    latest = pd.Timestamp(frame.stamp.max())
    rest_hours = float((kickoff - latest).total_seconds() / 3600.0)
    age_hours = (kickoff - frame.stamp).dt.total_seconds() / 3600.0
    last = frame.iloc[0]
    return {
        "matches": int(len(frame)),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "gf": gf,
        "ga": ga,
        "points": points,
        "avg_gf": float(frame.gf.mean()),
        "avg_ga": float(frame.ga.mean()),
        "gd": gf - ga,
        "over25_rate": float(over25),
        "btts_rate": float(btts),
        "last5": "".join(frame.result.astype(str).head(5).tolist()),
        "last_opponent": "",
        "last_competition": _safe_str(last["competition"]),
        "last_side": _safe_str(last["side"]),
        "rest_hours": rest_hours,
        "matches_3d": int((age_hours <= 72).sum()),
        "matches_7d": int((age_hours <= 168).sum()),
        "matches_14d": int((age_hours <= 336).sum()),
        "matches_30d": int((age_hours <= 720).sum()),
        "competitions_14d": int(frame.loc[age_hours <= 336, "competition"].replace("", np.nan).nunique()),
        "away_matches_14d": int((frame.loc[age_hours <= 336, "side"] == "away").sum()),
        "home_matches_14d": int((frame.loc[age_hours <= 336, "side"] == "home").sum()),
    }


def _summarize_h2h(
    events: list[dict[str, Any]],
    home_team_id: str,
    away_team_id: str,
    kickoff: pd.Timestamp,
) -> dict[str, Any]:
    rows = []
    home_wins = draws = away_wins = home_goals = away_goals = 0
    scorelines: list[str] = []
    for event in events:
        stamp = _ts(event.get("startTimestamp"))
        if stamp is None or stamp >= kickoff:
            continue
        home = event.get("homeTeam") or {}
        away = event.get("awayTeam") or {}
        hid = _safe_str(home.get("id"))
        aid = _safe_str(away.get("id"))
        if {hid, aid} != {_safe_str(home_team_id), _safe_str(away_team_id)}:
            continue
        hs = _safe_int((event.get("homeScore") or {}).get("current"))
        aws = _safe_int((event.get("awayScore") or {}).get("current"))
        if hs is None or aws is None:
            continue
        rows.append((stamp, hid, aid, hs, aws))
    rows.sort(key=lambda x: x[0], reverse=True)
    rows = rows[:MAX_H2H_EVENTS]
    for _, hid, aid, hs, aws in rows:
        oriented_home = hs if hid == _safe_str(home_team_id) else aws
        oriented_away = aws if hid == _safe_str(home_team_id) else hs
        home_goals += oriented_home
        away_goals += oriented_away
        if oriented_home > oriented_away:
            home_wins += 1
        elif oriented_home == oriented_away:
            draws += 1
        else:
            away_wins += 1
        scorelines.append(f"{oriented_home}-{oriented_away}")
    n = len(rows)
    return {
        "rich_h2h_matches": n if n else np.nan,
        "rich_h2h_home_wins": home_wins if n else np.nan,
        "rich_h2h_draws": draws if n else np.nan,
        "rich_h2h_away_wins": away_wins if n else np.nan,
        "rich_h2h_home_goals": home_goals if n else np.nan,
        "rich_h2h_away_goals": away_goals if n else np.nan,
        "rich_h2h_avg_total_goals": (
            float((home_goals + away_goals) / n) if n else np.nan
        ),
        "rich_h2h_btts_rate": (
            float(
                sum(
                    1
                    for _, _, _, hs, aws in rows
                    if hs > 0 and aws > 0
                )
                / n
            )
            if n
            else np.nan
        ),
        "rich_h2h_over25_rate": (
            float(sum(1 for _, _, _, hs, aws in rows if hs + aws >= 3) / n)
            if n
            else np.nan
        ),
        "rich_h2h_last5_scorelines": "|".join(scorelines[:5]),
    }


def _extract_team_ids_from_sofa_event(payload: dict[str, Any]) -> tuple[str, str]:
    home = _safe_str((payload.get("homeTeam") or {}).get("id"))
    away = _safe_str((payload.get("awayTeam") or {}).get("id"))
    return home, away


def _parse_sofa_standings(
    payload: dict[str, Any],
    home_team_id: str,
    away_team_id: str,
    kind: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for standing in payload.get("standings", []) or []:
        if not isinstance(standing, dict):
            continue
        for row in standing.get("rows", []) or []:
            if isinstance(row, dict):
                rows.append(row)

    out: dict[str, Any] = {}
    wanted = {_safe_str(home_team_id): "home", _safe_str(away_team_id): "away"}
    for row in rows:
        team_id = _safe_str((row.get("team") or {}).get("id"))
        side = wanted.get(team_id)
        if not side:
            continue
        prefix = f"rich_sofa_standing_{kind}_{side}_"
        for source, target in (
            ("position", "position"),
            ("matches", "matches"),
            ("wins", "wins"),
            ("draws", "draws"),
            ("losses", "losses"),
            ("scoresFor", "goals_for"),
            ("scoresAgainst", "goals_against"),
            ("points", "points"),
        ):
            value = _safe_int(row.get(source))
            if value is not None:
                out[prefix + target] = value
        score_diff = _safe_str(
            row.get("scoreDiffFormatted") or row.get("goalDifferenceFormatted")
        )
        if score_diff:
            out[prefix + "score_diff_formatted"] = score_diff
        promotion = row.get("promotion") or {}
        if isinstance(promotion, dict):
            text = _safe_str(promotion.get("text") or promotion.get("name"))
            if text:
                out[prefix + "promotion"] = text
        form = row.get("form")
        if isinstance(form, list):
            tokens = []
            for item in form:
                token = item if isinstance(item, str) else _safe_str(
                    (item or {}).get("result") or (item or {}).get("form")
                ) if isinstance(item, dict) else ""
                if token:
                    tokens.append(token)
            if tokens:
                out[prefix + "form"] = "|".join(tokens)
        elif isinstance(form, str) and form.strip():
            out[prefix + "form"] = form.strip()
    return out


def _blank_result() -> dict[str, Any]:
    return {
        "rich_detail_status": "NO_DETAIL",
        "rich_sources_observed": "",
        "rich_payload_count": 0,
        "rich_pit_status": "UNVERIFIABLE",
        "rich_pit_basis": (
            "current_retrieval_lower_bound_only;"
            "historical_publication_time_not_proven"
        ),
    }


def _fetch_and_enrich_row(
    row: dict[str, Any],
    *,
    fetcher: ExternalFetcher,
    now: pd.Timestamp,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    out = dict(row)
    kickoff = _ts(out.get("kickoff_utc"))
    diagnostics = _blank_result()
    raw_records: list[dict[str, Any]] = []
    if kickoff is None:
        diagnostics["rich_detail_status"] = "INVALID_KICKOFF"
        out.update(diagnostics)
        return out, raw_records

    hours = _hours_to_kickoff(kickoff, now)
    out["rich_hours_to_kickoff"] = hours
    if hours is not None and hours < -0.001:
        diagnostics["rich_detail_status"] = "PAST_FIXTURE"
        out.update(diagnostics)
        return out, raw_records

    observed_sources: set[str] = set()
    retrieval_times: list[pd.Timestamp] = []

    def remember(source: str, endpoint: str, payload: Any, retrieved_at: str) -> None:
        observed_sources.add(source)
        stamp = _ts(retrieved_at)
        if stamp is not None:
            retrieval_times.append(stamp)
        raw_records.append(
            {
                "match_id": _safe_str(out.get("match_id")),
                "provider_family": source,
                "endpoint": endpoint,
                "retrieved_at_utc": retrieved_at,
                "retrieval_before_kickoff": bool(
                    stamp is not None and stamp <= kickoff
                ),
                "payload_sha256": _json_hash(payload),
                "payload": payload,
            }
        )

    espn_event_id = _safe_str(out.get("espn_event_id"))
    competition = _safe_str(out.get("competition"))
    league = ESPN_LEAGUES.get(competition, "")
    if espn_event_id and league:
        try:
            summary, at = _get_json(
                fetcher,
                "espn_summary_rich",
                f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/summary",
                {"event": espn_event_id},
            )
            out.update(_parse_espn_summary(summary))
            out.update(_market_summary([summary]))
            remember("espn", "summary", summary, at)
        except Exception as exc:
            out["rich_espn_error"] = f"{type(exc).__name__}: {exc}"

        for side, team_key in (("home", "home_team_id"), ("away", "away_team_id")):
            team_id = _safe_str(out.get(team_key))
            if not team_id:
                continue
            try:
                schedule, at = _get_json(
                    fetcher,
                    "espn_team_schedule_rich",
                    f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/teams/{team_id}/schedule",
                    {"limit": 50},
                )
                recent = [
                    e for e in (schedule.get("events") or [])
                    if isinstance(e, dict)
                ]
                stats = _summarize_recent_events(recent, team_id, kickoff)
                prefix = f"rich_recent_{side}_"
                for key, value in stats.items():
                    out[prefix + key] = value

                if recent:
                    usable = []
                    for event in recent:
                        stamp = _event_start_timestamp(event.get("startTimestamp") or event.get("date"))
                        if stamp is None or stamp >= kickoff:
                            continue
                        competitors = (event.get("competitions") or [{}])[0]
                        parts = competitors.get("competitors") or []
                        opponent = next(
                            (
                                c for c in parts
                                if _safe_str(c.get("homeAway")) not in {"", side}
                            ),
                            None,
                        )
                        if isinstance(opponent, dict):
                            name = _safe_str(
                                (opponent.get("team") or {}).get("displayName")
                            )
                            if name:
                                usable.append((stamp, name))
                    if usable:
                        usable.sort(key=lambda x: x[0], reverse=True)
                        out[prefix + "last_opponent"] = usable[0][1]
                remember("espn", f"team_schedule:{team_id}", schedule, at)
            except Exception as exc:
                out[f"rich_recent_{side}_error"] = f"{type(exc).__name__}: {exc}"

    sofa_event_id = _safe_str(out.get("sofascore_event_id"))
    sofa_home_id = ""
    sofa_away_id = ""
    if sofa_event_id:
        try:
            detail, at = _get_json(
                fetcher,
                "sofascore_event_detail",
                f"https://api.sofascore.com/api/v1/event/{sofa_event_id}",
            )
            event_payload = detail.get("event") or detail
            out.update(_parse_sofa_event(event_payload))
            sofa_home_id, sofa_away_id = _extract_team_ids_from_sofa_event(event_payload)
            remember("sofascore", "event", detail, at)
        except Exception as exc:
            out["rich_sofa_event_error"] = f"{type(exc).__name__}: {exc}"

        try:
            lineups, at = _get_json(
                fetcher,
                "sofascore_lineups_rich",
                f"https://api.sofascore.com/api/v1/event/{sofa_event_id}/lineups",
            )
            out.update(_parse_lineups(lineups))
            remember("sofascore", "lineups", lineups, at)
        except Exception as exc:
            out["rich_sofa_lineups_error"] = f"{type(exc).__name__}: {exc}"

        try:
            managers, at = _get_json(
                fetcher,
                "sofascore_managers",
                f"https://api.sofascore.com/api/v1/event/{sofa_event_id}/managers",
            )
            out.update(
                {
                    "rich_sofa_home_manager": _safe_str(
                        (managers.get("home") or {}).get("manager", {}).get("name")
                        or (managers.get("home") or {}).get("name")
                    ),
                    "rich_sofa_away_manager": _safe_str(
                        (managers.get("away") or {}).get("manager", {}).get("name")
                        or (managers.get("away") or {}).get("name")
                    ),
                }
            )
            remember("sofascore", "managers", managers, at)
        except Exception as exc:
            out["rich_sofa_managers_error"] = f"{type(exc).__name__}: {exc}"

        sofa_tournament_id = _safe_str(
            (event_payload.get("tournament") or {}).get("uniqueTournament", {}).get("id")
            or (event_payload.get("uniqueTournament") or {}).get("id")
        )
        sofa_season_id = _safe_str((event_payload.get("season") or {}).get("id"))
        out["rich_sofa_unique_tournament_id"] = sofa_tournament_id
        out["rich_sofa_season_id"] = sofa_season_id

        if sofa_tournament_id and sofa_season_id and sofa_home_id and sofa_away_id:
            for kind in ("total", "home", "away"):
                try:
                    standings, at = _get_json(
                        fetcher,
                        f"sofascore_standings_{kind}",
                        f"https://api.sofascore.com/api/v1/unique-tournament/{sofa_tournament_id}/season/{sofa_season_id}/standings/{kind}",
                    )
                    out.update(
                        _parse_sofa_standings(
                            standings,
                            sofa_home_id,
                            sofa_away_id,
                            kind,
                        )
                    )
                    remember(
                        "sofascore",
                        f"standings/{kind}:{sofa_tournament_id}:{sofa_season_id}",
                        standings,
                        at,
                    )
                except Exception as exc:
                    out[f"rich_sofascore_standings_{kind}_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )

        try:
            pregame, at = _get_json(
                fetcher,
                "sofascore_pregame_form",
                f"https://api.sofascore.com/api/v1/event/{sofa_event_id}/pregame-form",
            )
            for key in ("homeTeam", "awayTeam", "home", "away"):
                value = pregame.get(key)
                if isinstance(value, dict):
                    for subkey in (
                        "form",
                        "score",
                        "position",
                        "goals",
                        "rating",
                    ):
                        if subkey in value and isinstance(
                            value[subkey], (str, int, float)
                        ):
                            out[f"rich_pregame_{key}_{subkey}"] = value[subkey]
            remember("sofascore", "pregame-form", pregame, at)
        except Exception as exc:
            out["rich_sofascore_pregame_error"] = f"{type(exc).__name__}: {exc}"

        if sofa_home_id and sofa_away_id:
            try:
                h2h, at = _get_json(
                    fetcher,
                    "sofascore_h2h",
                    f"https://api.sofascore.com/api/v1/event/{sofa_event_id}/h2h/events",
                )
                out.update(
                    _summarize_h2h(
                        h2h.get("events") or [],
                        sofa_home_id,
                        sofa_away_id,
                        kickoff,
                    )
                )
                remember("sofascore", "h2h/events", h2h, at)
            except Exception as exc:
                out["rich_sofascore_h2h_error"] = f"{type(exc).__name__}: {exc}"

            for side, team_id in (("home", sofa_home_id), ("away", sofa_away_id)):
                try:
                    last, at = _get_json(
                        fetcher,
                        "sofascore_team_last_events",
                        f"https://api.sofascore.com/api/v1/team/{team_id}/events/last/0",
                    )
                    stats = _summarize_recent_events(
                        last.get("events") or [],
                        team_id,
                        kickoff,
                    )
                    prefix = f"rich_sofa_recent_{side}_"
                    for key, value in stats.items():
                        out[prefix + key] = value
                    remember(
                        "sofascore",
                        f"team/{team_id}/events/last/0",
                        last,
                        at,
                    )
                except Exception as exc:
                    out[f"rich_sofa_recent_{side}_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )

    lat = _number(out.get("venue_lat"))
    lon = _number(out.get("venue_lon"))
    if lat is None or lon is None:
        location = ", ".join(
            x for x in (_safe_str(out.get("venue_city")), _safe_str(out.get("venue_country")))
            if x
        )
        if location:
            try:
                geo, at = _get_json(
                    fetcher,
                    "open_meteo_geocoding_rich",
                    "https://geocoding-api.open-meteo.com/v1/search",
                    {
                        "name": location,
                        "count": 1,
                        "language": "en",
                        "format": "json",
                    },
                )
                results = geo.get("results") or []
                if results:
                    lat = _number(results[0].get("latitude"))
                    lon = _number(results[0].get("longitude"))
                    out["rich_weather_geocoded_name"] = _safe_str(results[0].get("name"))
                    out["rich_weather_geocoded_country"] = _safe_str(results[0].get("country"))
                remember("open_meteo", "geocoding", geo, at)
            except Exception as exc:
                out["rich_weather_geocoding_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )

    if lat is not None and lon is not None:
        out["rich_weather_latitude"] = lat
        out["rich_weather_longitude"] = lon
        try:
            weather, at = _get_json(
                fetcher,
                "open_meteo_rich",
                "https://api.open-meteo.com/v1/forecast",
                {
                    "latitude": lat,
                    "longitude": lon,
                    "hourly": ",".join(
                        [
                            "temperature_2m",
                            "relative_humidity_2m",
                            "dew_point_2m",
                            "apparent_temperature",
                            "precipitation_probability",
                            "precipitation",
                            "rain",
                            "showers",
                            "snowfall",
                            "visibility",
                            "pressure_msl",
                            "surface_pressure",
                            "cloud_cover",
                            "cloud_cover_low",
                            "cloud_cover_mid",
                            "cloud_cover_high",
                            "wind_speed_10m",
                            "wind_direction_10m",
                            "wind_gusts_10m",
                            "is_day",
                            "sunshine_duration",
                            "cape",
                            "weather_code",
                        ]
                    ),
                    "timezone": "UTC",
                    "forecast_days": 2,
                },
            )
            out.update(_parse_weather_payload(weather, kickoff))
            remember("open_meteo", "forecast", weather, at)
        except Exception as exc:
            out["rich_weather_error"] = f"{type(exc).__name__}: {exc}"

    if retrieval_times:
        latest = max(retrieval_times)
        out["rich_last_retrieved_at_utc"] = latest.isoformat()
        out["rich_live_observable_before_kickoff"] = bool(latest <= kickoff)
    else:
        out["rich_last_retrieved_at_utc"] = ""
        out["rich_live_observable_before_kickoff"] = False

    diagnostics["rich_sources_observed"] = "|".join(sorted(observed_sources))
    diagnostics["rich_payload_count"] = int(len(raw_records))
    diagnostics["rich_detail_status"] = "ENRICHED" if raw_records else "NO_DETAIL"
    diagnostics["rich_snapshot_generated_at_utc"] = iso_utc(_now())

    error_fields = [
        key for key, value in out.items()
        if key.startswith("rich_") and key.endswith("_error") and _safe_str(value)
    ]
    core_rich_values = {
        key: value
        for key, value in out.items()
        if key.startswith("rich_")
        and not key.endswith("_error")
        and key not in {
            "rich_detail_status",
            "rich_sources_observed",
            "rich_payload_count",
            "rich_pit_status",
            "rich_pit_basis",
        }
    }
    populated = 0
    for value in core_rich_values.values():
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        populated += 1
    total = len(core_rich_values)
    diagnostics["rich_successful_endpoint_count"] = int(len(raw_records))
    diagnostics["rich_error_endpoint_count"] = int(len(error_fields))
    diagnostics["rich_source_family_count"] = int(len(observed_sources))
    diagnostics["rich_feature_field_count"] = int(populated)
    diagnostics["rich_feature_field_total"] = int(total)
    diagnostics["rich_information_completeness_ratio"] = (
        float(populated / total) if total else 0.0
    )
    diagnostics["rich_evidence_quality_state"] = (
        "HIGH" if len(raw_records) >= 8 and not error_fields
        else "MEDIUM" if raw_records
        else "LOW"
    )
    out.update(diagnostics)
    return out, raw_records


def enrich_matchday_frame(
    frame: pd.DataFrame,
    *,
    cache_dir: str = "cache/external",
    max_matches: int = DEFAULT_MAX_MATCHES,
    workers: int = DEFAULT_WORKERS,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    if frame.empty:
        return frame.copy(), {
            "status": "NO_FIXTURES",
            "input_rows": 0,
            "selected_rows": 0,
            "enriched_rows": 0,
            "raw_payload_records": 0,
            "historical_pit_claim": False,
            "production_changed": False,
        }, []

    work = frame.copy()
    work["kickoff_utc"] = pd.to_datetime(
        work["kickoff_utc"], utc=True, errors="coerce"
    )
    work = work.sort_values(
        ["kickoff_utc", "match_id"], kind="mergesort"
    ).reset_index(drop=True)
    now = pd.Timestamp(_now())

    eligible = work.loc[
        work["kickoff_utc"].notna() & (work["kickoff_utc"] >= now)
    ].copy()
    if max_matches > 0:
        eligible = eligible.head(int(max_matches))

    fetcher = ExternalFetcher(
        cache_dir=cache_dir,
        timeout=None,
        retries=8,
        backoff=2.0,
    )
    outputs: dict[int, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    errors: list[dict[str, Any]] = []

    if not eligible.empty:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = {
                pool.submit(
                    _fetch_and_enrich_row,
                    row.to_dict(),
                    fetcher=fetcher,
                    now=now,
                ): int(idx)
                for idx, row in eligible.iterrows()
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    outputs[idx] = future.result()
                except Exception as exc:
                    base = work.loc[idx].to_dict()
                    base.update(_blank_result())
                    base["rich_detail_status"] = "FAILED_ROW"
                    outputs[idx] = (base, [])
                    errors.append(
                        {
                            "match_id": _safe_str(base.get("match_id")),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

    rows: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    selected = set(outputs)
    for idx, row in work.iterrows():
        if idx in selected:
            enriched, records = outputs[idx]
            rows.append(enriched)
            raw_records.extend(records)
        else:
            untouched = row.to_dict()
            untouched.update(
                {
                    "rich_detail_status": "NOT_SELECTED",
                    "rich_pit_status": "UNVERIFIABLE",
                    "rich_pit_basis": (
                        "not_selected;current_retrieval_lower_bound_only"
                    ),
                }
            )
            rows.append(untouched)

    enriched = pd.DataFrame(rows)
    status = {
        "status": (
            "COMPLETED_RESEARCH_ONLY"
            if not errors
            else "COMPLETED_WITH_ROW_ERRORS"
        ),
        "input_rows": int(len(work)),
        "selected_rows": int(len(eligible)),
        "enriched_rows": int(len(outputs)),
        "raw_payload_records": int(len(raw_records)),
        "row_errors": errors,
        "workers": int(max(1, int(workers))),
        "max_matches": int(max_matches),
        "historical_pit_claim": False,
        "pit_basis": (
            "retrieval_time_is_recorded_but_source_publication_time_is_not_inferred"
        ),
        "production_changed": False,
    }
    return enriched, status, raw_records


def write_artifacts(
    frame: pd.DataFrame,
    status: dict[str, Any],
    raw_records: list[dict[str, Any]],
    *,
    output: str | Path,
    raw_output: str | Path,
    status_output: str | Path,
) -> None:
    output = Path(output)
    raw_output = Path(raw_output)
    status_output = Path(status_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    status_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    with raw_output.open("w", encoding="utf-8") as handle:
        for record in raw_records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                + "\\n"
            )
    status_output.write_text(
        json.dumps(status, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixtures",
        default="artifacts/future_matchday_fixtures.csv",
    )
    parser.add_argument(
        "--output",
        default="artifacts/matchday_rich_enrichment.csv",
    )
    parser.add_argument(
        "--raw-output",
        default="artifacts/matchday_rich_raw.jsonl",
    )
    parser.add_argument(
        "--status",
        default="artifacts/matchday_rich_enrichment_status.json",
    )
    parser.add_argument("--cache-dir", default="cache/external")
    parser.add_argument("--max-matches", type=int, default=DEFAULT_MAX_MATCHES)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    if args.max_matches < 0:
        parser.error("--max-matches must be >= 0")
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    path = Path(args.fixtures)
    try:
        frame = pd.read_csv(path)
    except FileNotFoundError:
        status = {
            "status": "FAILED_INPUT_MISSING",
            "fixtures": str(path),
            "production_changed": False,
        }
        Path(args.status).parent.mkdir(parents=True, exist_ok=True)
        Path(args.status).write_text(
            json.dumps(status, indent=2),
            encoding="utf-8",
        )
        return 1
    except Exception as exc:
        status = {
            "status": "FAILED_INPUT_READ",
            "error": f"{type(exc).__name__}: {exc}",
            "production_changed": False,
        }
        Path(args.status).parent.mkdir(parents=True, exist_ok=True)
        Path(args.status).write_text(
            json.dumps(status, indent=2),
            encoding="utf-8",
        )
        return 1

    try:
        enriched, status, raw_records = enrich_matchday_frame(
            frame,
            cache_dir=args.cache_dir,
            max_matches=args.max_matches,
            workers=args.workers,
        )
        write_artifacts(
            enriched,
            status,
            raw_records,
            output=args.output,
            raw_output=args.raw_output,
            status_output=args.status,
        )
    except Exception as exc:
        status = {
            "status": "FAILED_INTERNAL",
            "error": f"{type(exc).__name__}: {exc}",
            "production_changed": False,
        }
        Path(args.status).parent.mkdir(parents=True, exist_ok=True)
        Path(args.status).write_text(
            json.dumps(status, indent=2),
            encoding="utf-8",
        )
        return 1
    print(json.dumps(status, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
