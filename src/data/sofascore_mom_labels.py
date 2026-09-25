"""SofaScore MOM-label acquisition for research-only training labels.

This module never turns a missing label into a negative. It matches historical
fixtures to SofaScore events using team identity + kickoff proximity, then reads
only the post-match `playerOfTheMatch` label. Labels are kept separate from
pre-match feature construction.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
import unicodedata
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.data.external_fetch import ExternalFetcher


SOFASCORE_BASE = "https://api.sofascore.com/api/v1"
SOFASCORE_FALLBACK_BASE = "https://www.sofascore.com/api/v1"
SOFASCORE_HEADERS = {
    "User-Agent": "SoccerPredictionResearch/1.0 MOM-label-audit",
    "Referer": "https://www.sofascore.com/",
}
DEFAULT_MAX_EVENT_DELTA_HOURS = 6.0


def _norm_team(value: Any) -> str:
    # NFKD removes compatibility noise while preserving non-Latin identity.
    # Strip Latin combining marks only; Japanese kana dakuten/handakuten must
    # survive because they are part of the team name's identity.
    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    kept: list[str] = []
    for ch in decomposed:
        if unicodedata.combining(ch):
            previous = kept[-1] if kept else ""
            if "LATIN" in unicodedata.name(previous, ""):
                continue
        kept.append(ch)
    text = unicodedata.normalize("NFKC", "".join(kept).casefold())
    # Remove punctuation/separators without treating non-Latin scripts as
    # word separators. Do not use fuzzy similarity.
    return "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] not in {"P", "Z"}
    )


def _parse_json(body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"SofaScore JSON decode failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("SofaScore response root must be an object")
    return payload


def _extract_player_of_match(payload: dict[str, Any]) -> tuple[str, str]:
    candidate = payload.get("playerOfTheMatch")
    if isinstance(candidate, dict) and isinstance(candidate.get("player"), dict):
        candidate = candidate["player"]
    if not isinstance(candidate, dict):
        return "", ""
    player_id = candidate.get("id")
    name = candidate.get("name") or candidate.get("shortName") or ""
    if player_id in (None, ""):
        return "", ""
    return str(player_id), str(name)


def _utc_ts(value: Any) -> pd.Timestamp | None:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(ts) else ts


def _event_candidates(events: list[dict[str, Any]], fixture: pd.Series, max_delta_hours: float) -> list[dict[str, Any]]:
    home = _norm_team(fixture.get("home_team"))
    away = _norm_team(fixture.get("away_team"))
    kickoff = _utc_ts(fixture.get("kickoff_utc"))
    if not home or not away or kickoff is None:
        return []

    out: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        home_obj = event.get("homeTeam") if isinstance(event.get("homeTeam"), dict) else {}
        away_obj = event.get("awayTeam") if isinstance(event.get("awayTeam"), dict) else {}
        if _norm_team(home_obj.get("name")) != home or _norm_team(away_obj.get("name")) != away:
            continue
        start = event.get("startTimestamp")
        if start in (None, ""):
            continue
        try:
            event_time = pd.to_datetime(int(start), unit="s", utc=True)
        except (TypeError, ValueError, OverflowError):
            continue
        delta_hours = abs(float((event_time - kickoff).total_seconds()) / 3600.0)
        if delta_hours <= float(max_delta_hours):
            out.append({
                "event": event,
                "delta_hours": delta_hours,
                "event_time_utc": event_time,
            })
    out.sort(key=lambda x: (x["delta_hours"], int(x["event"].get("id") or 0)))
    return out


def fetch_scheduled_events_for_date(
    date_utc: str,
    *,
    fetcher: ExternalFetcher,
) -> tuple[list[dict[str, Any]], str]:
    url = f"{SOFASCORE_BASE}/sport/football/scheduled-events/{date_utc}"
    response = fetcher.get(
        "sofascore_scheduled_events",
        url,
        headers=SOFASCORE_HEADERS,
    )
    payload = _parse_json(response.body)
    events = payload.get("events")
    if not isinstance(events, list):
        raise RuntimeError("SofaScore scheduled-events payload missing events list")
    return [e for e in events if isinstance(e, dict)], response.metadata.retrieved_at


def fetch_mom_label(
    event_id: str | int,
    *,
    fetcher: ExternalFetcher,
) -> dict[str, Any]:
    url = f"{SOFASCORE_BASE}/event/{int(event_id)}/best-players/summary"
    response = fetcher.get(
        "sofascore_best_players_summary",
        url,
        headers=SOFASCORE_HEADERS,
    )
    payload = _parse_json(response.body)
    player_id, player_name = _extract_player_of_match(payload)
    return {
        "event_id": str(int(event_id)),
        "player_id": player_id,
        "player_name": player_name,
        "label_found": bool(player_id),
        "retrieved_at_utc": response.metadata.retrieved_at,
        "source_content_sha256": hashlib.sha256(response.body).hexdigest(),
        "source": "sofascore_best_players_summary",
        "cache_hit": bool(getattr(response.metadata, "cache_hit", False)),
    }


def _extract_unique_tournaments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            items = value.get("uniqueTournaments")
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    try:
                        tid = int(item["id"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    name = str(item.get("name") or "").strip()
                    if name and (tid, name) not in seen:
                        seen.add((tid, name))
                        found.append(item)
            for child in value.values():
                if isinstance(child, (dict, list)):
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    return found


def fetch_unique_football_tournaments(*, fetcher: ExternalFetcher) -> tuple[list[dict[str, Any]], str]:
    endpoints = (
        f"{SOFASCORE_BASE}/sport/football/unique-tournaments",
        f"{SOFASCORE_BASE}/config/unique-tournaments/en/football",
        f"{SOFASCORE_FALLBACK_BASE}/sport/football/unique-tournaments",
    )
    errors: list[str] = []
    for url in endpoints:
        try:
            response = fetcher.get("sofascore_unique_football_tournaments", url, headers=SOFASCORE_HEADERS)
            tournaments = _extract_unique_tournaments(_parse_json(response.body))
            if tournaments:
                return tournaments, response.metadata.retrieved_at
            errors.append(f"{url}: tournament list empty")
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    raise RuntimeError("SofaScore football tournament discovery failed on all public endpoints: " + " | ".join(errors))


def _sample_discovery_dates(dates_utc: list[str], *, max_dates: int = 5) -> list[str]:
    """Choose deterministic, well-spread dates for event-based tournament discovery."""
    values = sorted({str(x)[:10] for x in dates_utc if str(x).strip()})
    if not values:
        raise ValueError("at least one discovery date is required")
    limit = max(2, int(max_dates))
    if len(values) <= limit:
        return values
    indices = np.linspace(0, len(values) - 1, num=limit, dtype=int).tolist()
    return [values[i] for i in indices]


def discover_unique_tournament_from_scheduled_events(
    dates_utc: list[str],
    *,
    names: list[str] | tuple[str, ...],
    fetcher: ExternalFetcher,
    category_names: list[str] | tuple[str, ...] = (),
    max_dates: int = 5,
) -> tuple[int, dict[str, Any]]:
    """Resolve a tournament id from exact uniqueTournament objects in date-scoped events.

    The registry-wide tournament endpoint can be unavailable in hosted runners.
    This fallback uses real scheduled-event payloads on deterministic sample dates,
    requires exact normalized tournament-name/slug matching, and accepts an id only
    when the same id is observed on at least two independent dates.
    """
    requested = {_norm_team(name) for name in names if str(name).strip()}
    category_requested = {_norm_team(name) for name in category_names if str(name).strip()}
    if not requested:
        raise ValueError("at least one tournament name is required")
    sample_dates = _sample_discovery_dates(dates_utc, max_dates=max_dates)

    observations: list[dict[str, Any]] = []
    errors: list[str] = []
    for date_utc in sample_dates:
        try:
            events, retrieved_at = fetch_scheduled_events_for_date(date_utc, fetcher=fetcher)
        except Exception as exc:
            errors.append(f"{date_utc}: {type(exc).__name__}: {exc}")
            continue

        matches: list[tuple[int, str, str]] = []
        for event in events:
            tournament = event.get("uniqueTournament")
            if not isinstance(tournament, dict):
                continue
            raw_name = str(tournament.get("name") or "").strip()
            raw_slug = str(tournament.get("slug") or "").strip()
            norm_name = _norm_team(raw_name)
            norm_slug = _norm_team(raw_slug)
            if norm_name not in requested and norm_slug not in requested:
                continue
            try:
                tid = int(tournament["id"])
            except (KeyError, TypeError, ValueError):
                continue
            category = tournament.get("category")
            category_name = (
                _norm_team(category.get("name"))
                if isinstance(category, dict)
                else ""
            )
            matches.append((tid, raw_name, category_name))

        by_id: dict[int, tuple[str, str]] = {}
        for tid, raw_name, category_name in matches:
            if tid not in by_id:
                by_id[tid] = (raw_name, category_name)
        if len(by_id) == 1:
            tid, (raw_name, category_name) = next(iter(by_id.items()))
            observations.append({
                "date_utc": date_utc,
                "tournament_id": tid,
                "name": raw_name,
                "category": category_name,
                "retrieved_at_utc": retrieved_at,
            })

    counts = pd.Series([x["tournament_id"] for x in observations], dtype="int64").value_counts()
    if counts.empty:
        detail = " | ".join(errors) if errors else "no exact tournament observations"
        raise RuntimeError(
            "SofaScore event-based tournament discovery failed: "
            f"dates={sample_dates}; {detail}"
        )

    candidate_ids = [int(tid) for tid, count in counts.items() if int(count) >= 2]
    # Singleton competitions (for example a one-match super cup) can only
    # provide one independent fixture date. Accept that case only when the
    # exact tournament name/slug resolves to one id and no competing id exists.
    if not candidate_ids and len(sample_dates) == 1 and len(counts) == 1:
        candidate_ids = [int(counts.index[0])]
    if category_requested and len(candidate_ids) > 1:
        preferred = [
            tid for tid in candidate_ids
            if any(
                obs["tournament_id"] == tid and obs["category"] in category_requested
                for obs in observations
            )
        ]
        if preferred:
            candidate_ids = preferred
    if len(candidate_ids) != 1:
        raise RuntimeError(
            "SofaScore event-based tournament discovery is ambiguous or insufficiently repeated: "
            f"observations={observations}"
        )

    tournament_id = candidate_ids[0]
    used = [x for x in observations if int(x["tournament_id"]) == tournament_id]
    return tournament_id, {
        "method": "SCHEDULED_EVENTS_UNIQUE_TOURNAMENT",
        "sample_dates_utc": sample_dates,
        "observations": used,
        "independent_date_count": len(used),
        "category_filter": sorted(category_requested),
    }


def resolve_unique_tournament_id(
    tournaments: list[dict[str, Any]],
    *,
    names: list[str] | tuple[str, ...],
    category_names: list[str] | tuple[str, ...] = (),
) -> int:
    requested = {_norm_team(name) for name in names if str(name).strip()}
    category_requested = {_norm_team(name) for name in category_names if str(name).strip()}
    if not requested:
        raise ValueError("at least one tournament name is required")
    candidates: list[dict[str, Any]] = []
    for tournament in tournaments:
        tid = tournament.get("id")
        try:
            int(tid)
        except (TypeError, ValueError):
            continue
        if _norm_team(tournament.get("name")) in requested or _norm_team(tournament.get("slug")) in requested:
            candidates.append(tournament)
    if category_requested and len(candidates) > 1:
        preferred = []
        for tournament in candidates:
            category = tournament.get("category")
            category_name = _norm_team(category.get("name")) if isinstance(category, dict) else ""
            if category_name in category_requested:
                preferred.append(tournament)
        if preferred:
            candidates = preferred
    ids = sorted({int(x["id"]) for x in candidates})
    if len(ids) != 1:
        raise RuntimeError(f"SofaScore tournament lookup is ambiguous or missing: names={sorted(requested)} ids={ids}")
    return ids[0]


def discover_unique_season_from_scheduled_events(
    dates_utc: list[str],
    *,
    tournament_id: int,
    season_start_year: int,
    fetcher: ExternalFetcher,
    max_dates: int = 5,
) -> tuple[int, dict[str, Any]]:
    """Resolve a season id from exact scheduled-event tournament/season objects."""
    sample_dates = _sample_discovery_dates(dates_utc, max_dates=max_dates)
    wanted_year = str(int(season_start_year))
    observations: list[dict[str, Any]] = []
    errors: list[str] = []

    for date_utc in sample_dates:
        try:
            events, retrieved_at = fetch_scheduled_events_for_date(date_utc, fetcher=fetcher)
        except Exception as exc:
            errors.append(f"{date_utc}: {type(exc).__name__}: {exc}")
            continue

        ids_seen_on_date: set[int] = set()
        for event in events:
            tournament = event.get("uniqueTournament")
            if not isinstance(tournament, dict):
                continue
            try:
                event_tournament_id = int(tournament.get("id"))
            except (TypeError, ValueError):
                continue
            if event_tournament_id != int(tournament_id):
                continue
            season = event.get("season")
            if not isinstance(season, dict):
                continue
            raw_name = str(season.get("name") or "").strip()
            raw_year = str(season.get("year") or "").strip()
            if wanted_year not in raw_name and raw_year != wanted_year:
                continue
            try:
                season_id = int(season["id"])
            except (KeyError, TypeError, ValueError):
                continue
            ids_seen_on_date.add(season_id)
        if len(ids_seen_on_date) == 1:
            season_id = next(iter(ids_seen_on_date))
            observations.append({
                "date_utc": date_utc,
                "season_id": season_id,
                "season_start_year": int(season_start_year),
                "retrieved_at_utc": retrieved_at,
            })

    counts = pd.Series([x["season_id"] for x in observations], dtype="int64").value_counts()
    candidate_ids = [int(season_id) for season_id, count in counts.items() if int(count) >= 2]
    if not candidate_ids and len(sample_dates) == 1 and len(counts) == 1:
        candidate_ids = [int(counts.index[0])]
    if len(candidate_ids) != 1:
        detail = " | ".join(errors) if errors else "no exact season observations"
        raise RuntimeError(
            "SofaScore event-based season discovery is ambiguous or insufficiently repeated: "
            f"tournament={int(tournament_id)} season_start_year={int(season_start_year)} "
            f"sample_dates={sample_dates} observations={observations}; {detail}"
        )
    season_id = candidate_ids[0]
    used = [x for x in observations if int(x["season_id"]) == season_id]
    return season_id, {
        "method": "SCHEDULED_EVENTS_SEASON_FALLBACK",
        "sample_dates_utc": sample_dates,
        "observations": used,
        "independent_date_count": len(used),
    }


def fetch_unique_tournament_seasons(
    tournament_id: int,
    *,
    fetcher: ExternalFetcher,
) -> tuple[list[dict[str, Any]], str]:
    url = f"{SOFASCORE_BASE}/unique-tournament/{int(tournament_id)}/seasons"
    response = fetcher.get(
        "sofascore_tournament_seasons",
        url,
        headers=SOFASCORE_HEADERS,
    )
    payload = _parse_json(response.body)
    seasons = payload.get("seasons")
    if not isinstance(seasons, list):
        raise RuntimeError("SofaScore seasons payload missing seasons list")
    return [x for x in seasons if isinstance(x, dict)], response.metadata.retrieved_at


def select_season_id(
    seasons: list[dict[str, Any]],
    season_start_year: int,
) -> int:
    matches = []
    for season in seasons:
        raw_name = str(season.get("name") or season.get("year") or "")
        if str(int(season_start_year)) in raw_name:
            try:
                matches.append((int(season["id"]), raw_name))
            except (TypeError, ValueError):
                continue
    if len(matches) != 1:
        raise RuntimeError(
            f"SofaScore season lookup is ambiguous for {season_start_year}: {matches}"
        )
    return matches[0][0]


def _extract_events_list(payload: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Accept known SofaScore wrappers without weakening fail-closed semantics."""
    direct = payload.get("events")
    if isinstance(direct, list):
        return [x for x in direct if isinstance(x, dict)]
    nested = payload.get("data")
    if isinstance(nested, dict) and isinstance(nested.get("events"), list):
        return [x for x in nested["events"] if isinstance(x, dict)]
    return None


def fetch_scheduled_events_for_dates(
    dates_utc: list[str],
    *,
    fetcher: ExternalFetcher,
    request_delay_seconds: float = 0.0,
) -> tuple[list[dict[str, Any]], str]:
    """Build a complete date-scoped event index, failing closed on any date fetch error."""
    delay = float(request_delay_seconds)
    if not math.isfinite(delay) or delay < 0:
        raise ValueError("request_delay_seconds must be finite and non-negative")
    events: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    retrieved: list[str] = []
    for idx, date_utc in enumerate(sorted(set(dates_utc))):
        event_list, retrieved_at = fetch_scheduled_events_for_date(date_utc, fetcher=fetcher)
        retrieved.append(retrieved_at)
        for event in event_list:
            event_id = str(event.get("id") or "").strip()
            if event_id and event_id in seen_ids:
                continue
            if event_id:
                seen_ids.add(event_id)
            events.append(event)
        if delay > 0 and idx + 1 < len(set(dates_utc)):
            time.sleep(delay)
    if not events:
        raise RuntimeError("SofaScore scheduled-events fallback returned no events")
    return events, max(retrieved)


def fetch_tournament_season_events(
    tournament_id: int,
    season_id: int,
    *,
    fetcher: ExternalFetcher,
    max_pages: int = 200,
) -> tuple[list[dict[str, Any]], str]:
    events: list[dict[str, Any]] = []
    retrieval_times: list[str] = []
    seen_ids: set[str] = set()
    for page in range(max(1, int(max_pages))):
        payload = None
        response = None
        errors: list[str] = []
        for base in (SOFASCORE_BASE, SOFASCORE_FALLBACK_BASE):
            url = f"{base}/unique-tournament/{int(tournament_id)}/season/{int(season_id)}/events/last/{page}"
            try:
                candidate = fetcher.get(
                    "sofascore_tournament_season_events",
                    url,
                    headers=SOFASCORE_HEADERS,
                )
                parsed = _parse_json(candidate.body)
                page_events = _extract_events_list(parsed)
                if isinstance(page_events, list):
                    response = candidate
                    payload = parsed
                    break
                errors.append(f"{base}: payload missing events list")
            except Exception as exc:
                errors.append(f"{base}: {type(exc).__name__}: {exc}")
        if response is None or payload is None:
            raise RuntimeError(
                "SofaScore season events acquisition failed on all public hosts: "
                + " | ".join(errors)
            )
        retrieval_times.append(response.metadata.retrieved_at)
        page_events = _extract_events_list(payload)
        if not isinstance(page_events, list):
            raise RuntimeError("SofaScore season events payload missing events list")
        if not page_events:
            break
        added = 0
        for event in page_events:
            if not isinstance(event, dict) or event.get("id") in (None, ""):
                continue
            event_id = str(event["id"])
            if event_id in seen_ids:
                continue
            seen_ids.add(event_id)
            events.append(event)
            added += 1
        if added == 0:
            break
    if not events:
        raise RuntimeError(
            f"SofaScore season event acquisition returned no events for tournament={tournament_id} season={season_id}"
        )
    return events, max(retrieval_times)


def fetch_tournament_season_events_by_rounds(
    tournament_id: int,
    season_id: int,
    *,
    fetcher: ExternalFetcher,
    max_rounds: int = 50,
    request_delay_seconds: float = 1.5,
) -> tuple[list[dict[str, Any]], str]:
    """Fallback for seasons whose paginated events endpoint is unavailable.

    Each numbered round is acquired independently. Any failed round aborts the
    fallback rather than silently accepting partial coverage.
    """
    rounds = max(1, int(max_rounds))
    delay = float(request_delay_seconds)
    if not math.isfinite(delay) or delay < 0:
        raise ValueError("request_delay_seconds must be finite and non-negative")

    events: list[dict[str, Any]] = []
    retrieval_times: list[str] = []
    seen_ids: set[str] = set()
    consecutive_empty = 0

    for round_number in range(1, rounds + 1):
        errors: list[str] = []
        round_events: list[dict[str, Any]] | None = None
        response = None
        for base in (SOFASCORE_BASE, SOFASCORE_FALLBACK_BASE):
            url = (
                f"{base}/unique-tournament/{int(tournament_id)}/season/"
                f"{int(season_id)}/events/round/{int(round_number)}"
            )
            try:
                candidate = fetcher.get(
                    "sofascore_tournament_round_events",
                    url,
                    headers=SOFASCORE_HEADERS,
                )
                parsed = _parse_json(candidate.body)
                extracted = _extract_events_list(parsed)
                if isinstance(extracted, list):
                    round_events = extracted
                    response = candidate
                    break
                errors.append(f"{base}: payload missing events list")
            except Exception as exc:
                errors.append(f"{base}: {type(exc).__name__}: {exc}")
        if response is None or round_events is None:
            raise RuntimeError(
                f"SofaScore round {round_number} acquisition failed on all public hosts: "
                + " | ".join(errors)
            )

        retrieval_times.append(response.metadata.retrieved_at)
        if not round_events:
            consecutive_empty += 1
            if events and consecutive_empty >= 3:
                break
        else:
            consecutive_empty = 0
            for event in round_events:
                if not isinstance(event, dict) or event.get("id") in (None, ""):
                    continue
                event_id = str(event["id"])
                if event_id in seen_ids:
                    continue
                seen_ids.add(event_id)
                events.append(event)

        if delay > 0 and round_number < rounds:
            time.sleep(delay)

    if not events:
        raise RuntimeError(
            f"SofaScore round event acquisition returned no events for "
            f"tournament={tournament_id} season={season_id}"
        )
    return events, max(retrieval_times)


def collect_sofascore_mom_labels_tournament_season(
    fixtures: pd.DataFrame,
    *,
    tournament_id: int,
    season_id: int,
    cache_dir: str = "cache/external",
    max_event_delta_hours: float = DEFAULT_MAX_EVENT_DELTA_HOURS,
    retries: int = 3,
    max_event_pages: int = 200,
    request_delay_seconds: float = 1.5,
) -> pd.DataFrame:
    required = {"match_id", "kickoff_utc", "home_team", "away_team"}
    missing = sorted(required - set(fixtures.columns))
    if missing:
        raise ValueError(f"MOM label fixtures missing columns: {missing}")

    d = fixtures.copy()
    d["match_id"] = d["match_id"].astype("string").str.strip()
    d["kickoff_utc"] = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    if d["kickoff_utc"].isna().any() or d["match_id"].eq("").any():
        raise ValueError("MOM label fixtures contain invalid match_id/kickoff_utc")

    delay = float(request_delay_seconds)
    if not math.isfinite(delay) or delay < 0:
        raise ValueError("request_delay_seconds must be finite and non-negative")
    fetcher = ExternalFetcher(cache_dir=cache_dir, retries=retries)
    try:
        events, _ = fetch_tournament_season_events(
            tournament_id,
            season_id,
            fetcher=fetcher,
            max_pages=max_event_pages,
        )
        event_source = "tournament_season_events"
    except RuntimeError as primary_exc:
        try:
            events, _ = fetch_tournament_season_events_by_rounds(
                tournament_id,
                season_id,
                fetcher=fetcher,
                max_rounds=50,
                request_delay_seconds=delay,
            )
            event_source = "tournament_round_events_fallback"
        except Exception as round_exc:
            # The season and round endpoints can both be unavailable on some
            # public hosts. Keep the existing date fallback as the last route;
            # if any required date is unavailable, fail closed.
            fallback_dates = sorted(
                {
                    (ts - pd.Timedelta(days=1)).date().isoformat()
                    for ts in d["kickoff_utc"]
                }
                | {
                    ts.date().isoformat()
                    for ts in d["kickoff_utc"]
                }
                | {
                    (ts + pd.Timedelta(days=1)).date().isoformat()
                    for ts in d["kickoff_utc"]
                }
            )
            try:
                events, _ = fetch_scheduled_events_for_dates(
                    fallback_dates,
                    fetcher=fetcher,
                    request_delay_seconds=delay,
                )
                event_source = "scheduled_events_fallback"
            except Exception as fallback_exc:
                raise RuntimeError(
                    "SofaScore MOM event acquisition failed via season, round, and "
                    "scheduled-date paths: "
                    f"season={primary_exc}; round={type(round_exc).__name__}: {round_exc}; "
                    f"fallback={type(fallback_exc).__name__}: {fallback_exc}"
                ) from fallback_exc

    rows: list[dict[str, Any]] = []
    for fixture in d.itertuples(index=False):
        series = pd.Series(fixture._asdict())
        candidates = _event_candidates(events, series, max_event_delta_hours)
        if not candidates:
            rows.append({
                "match_id": str(series["match_id"]),
                "label_status": "EVENT_NOT_MATCHED",
                "event_id": "",
                "player_id": "",
                "player_name": "",
                "label_source": "sofascore_best_players_summary",
                "label_retrieved_at_utc": "",
                "event_kickoff_utc": "",
                "event_match_delta_hours": None,
                "source_content_sha256": "",
            })
            continue
        best = candidates[0]
        tie = [
            x for x in candidates
            if abs(float(x["delta_hours"]) - float(best["delta_hours"])) < 1e-9
        ]
        if len(tie) > 1:
            rows.append({
                "match_id": str(series["match_id"]),
                "label_status": "AMBIGUOUS_EVENT_MATCH",
                "event_id": "",
                "player_id": "",
                "player_name": "",
                "label_source": "sofascore_best_players_summary",
                "label_retrieved_at_utc": "",
                "event_kickoff_utc": "",
                "event_match_delta_hours": float(best["delta_hours"]),
                "source_content_sha256": "",
            })
            continue

        event_id = best["event"].get("id")
        if event_id in (None, ""):
            raise RuntimeError(f"SofaScore matched event has no id for match_id={series['match_id']!r}")
        label = fetch_mom_label(event_id, fetcher=fetcher)
        if not label.get("cache_hit", False) and delay > 0:
            time.sleep(delay)
        event_time = best.get("event_kickoff_utc")
        rows.append({
            "match_id": str(series["match_id"]),
            "label_status": "LABEL_FOUND" if label["label_found"] else "LABEL_MISSING",
            "event_id": label["event_id"],
            "player_id": label["player_id"],
            "player_name": label["player_name"],
            "label_source": label["source"],
            "label_retrieved_at_utc": label["retrieved_at_utc"],
            "event_kickoff_utc": event_time.isoformat() if isinstance(event_time, pd.Timestamp) else "",
            "event_match_delta_hours": float(best["delta_hours"]),
            "source_content_sha256": label["source_content_sha256"],
        })

    out = pd.DataFrame(rows)
    if not out.empty and out["match_id"].duplicated().any():
        raise RuntimeError("MOM label acquisition produced duplicate match_id rows")
    return out

def collect_sofascore_mom_labels(
    fixtures: pd.DataFrame,
    *,
    cache_dir: str = "cache/external",
    max_event_delta_hours: float = DEFAULT_MAX_EVENT_DELTA_HOURS,
    retries: int = 3,
) -> pd.DataFrame:
    required = {"match_id", "kickoff_utc", "home_team", "away_team"}
    missing = sorted(required - set(fixtures.columns))
    if missing:
        raise ValueError(f"MOM label fixtures missing columns: {missing}")

    d = fixtures.copy()
    d["match_id"] = d["match_id"].astype("string").str.strip()
    d["kickoff_utc"] = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    if d["kickoff_utc"].isna().any() or d["match_id"].eq("").any():
        raise ValueError("MOM label fixtures contain invalid match_id/kickoff_utc")

    fetcher = ExternalFetcher(cache_dir=cache_dir, retries=retries)
    dates = sorted(
        {
            (ts - pd.Timedelta(days=1)).date().isoformat()
            for ts in d["kickoff_utc"]
        }
        | {
            ts.date().isoformat()
            for ts in d["kickoff_utc"]
        }
        | {
            (ts + pd.Timedelta(days=1)).date().isoformat()
            for ts in d["kickoff_utc"]
        }
    )

    event_index: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    for date_utc in dates:
        events, retrieved_at = fetch_scheduled_events_for_date(date_utc, fetcher=fetcher)
        for event in events:
            event_id = str(event.get("id") or "").strip()
            # Scheduled-events endpoints can overlap adjacent-date windows. Deduplicate
            # by provider event identity so one real fixture cannot become ambiguous.
            if event_id and event_id in seen_event_ids:
                continue
            if event_id:
                seen_event_ids.add(event_id)
            event_index.append({
                "event": event,
                "scheduled_retrieved_at_utc": retrieved_at,
            })

    rows: list[dict[str, Any]] = []
    for fixture in d.itertuples(index=False):
        series = pd.Series(fixture._asdict())
        candidates = _event_candidates(
            [x["event"] for x in event_index],
            series,
            max_event_delta_hours,
        )
        if not candidates:
            rows.append({
                "match_id": str(series["match_id"]),
                "label_status": "EVENT_NOT_MATCHED",
                "event_id": "",
                "player_id": "",
                "player_name": "",
                "label_source": "sofascore_best_players_summary",
                "label_retrieved_at_utc": "",
                "event_kickoff_utc": "",
                "event_match_delta_hours": None,
                "source_content_sha256": "",
            })
            continue
        best = candidates[0]
        tie = [x for x in candidates if abs(float(x["delta_hours"]) - float(best["delta_hours"])) < 1e-9]
        if len(tie) > 1:
            rows.append({
                "match_id": str(series["match_id"]),
                "label_status": "AMBIGUOUS_EVENT_MATCH",
                "event_id": "",
                "player_id": "",
                "player_name": "",
                "label_source": "sofascore_best_players_summary",
                "label_retrieved_at_utc": "",
                "event_kickoff_utc": "",
                "event_match_delta_hours": float(best["delta_hours"]),
                "source_content_sha256": "",
            })
            continue

        event_id = best["event"].get("id")
        if event_id in (None, ""):
            raise RuntimeError(f"SofaScore matched event has no id for match_id={series['match_id']!r}")
        label = fetch_mom_label(event_id, fetcher=fetcher)
        event_time = best.get("event_kickoff_utc")
        event_time_utc = event_time.isoformat() if isinstance(event_time, pd.Timestamp) else ""
        rows.append({
            "match_id": str(series["match_id"]),
            "label_status": "LABEL_FOUND" if label["label_found"] else "LABEL_MISSING",
            "event_id": label["event_id"],
            "player_id": label["player_id"],
            "player_name": label["player_name"],
            "label_source": label["source"],
            "label_retrieved_at_utc": label["retrieved_at_utc"],
            "event_kickoff_utc": event_time_utc,
            "event_match_delta_hours": float(best["delta_hours"]),
            "source_content_sha256": label["source_content_sha256"],
        })

    out = pd.DataFrame(rows)
    if not out.empty and out["match_id"].duplicated().any():
        raise RuntimeError("MOM label acquisition produced duplicate match_id rows")
    return out


def label_data_contract_report(labels: pd.DataFrame) -> dict[str, Any]:
    """Report label coverage without converting missing labels to negatives."""
    if labels.empty:
        return {
            "status": "DEFERRED_NO_MOM_LABELS",
            "matches": 0,
            "label_found": 0,
            "fail_closed": True,
        }
    status = labels["label_status"].astype("string")
    found = status.eq("LABEL_FOUND")
    return {
        "status": "READY_FOR_JOIN" if bool(found.any()) else "DEFERRED_NO_VERIFIED_MOM_LABELS",
        "matches": int(labels["match_id"].nunique()),
        "label_found": int(found.sum()),
        "label_missing": int(status.eq("LABEL_MISSING").sum()),
        "event_not_matched": int(status.eq("EVENT_NOT_MATCHED").sum()),
        "ambiguous_event_match": int(status.eq("AMBIGUOUS_EVENT_MATCH").sum()),
        "fail_closed": True,
    }
