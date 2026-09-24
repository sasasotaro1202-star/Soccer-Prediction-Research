"""SofaScore MOM-label acquisition for research-only training labels.

This module never turns a missing label into a negative. It matches historical
fixtures to SofaScore events using team identity + kickoff proximity, then reads
only the post-match `playerOfTheMatch` label. Labels are kept separate from
pre-match feature construction.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.data.external_fetch import ExternalFetcher


SOFASCORE_BASE = "https://api.sofascore.com/api/v1"
SOFASCORE_HEADERS = {
    "User-Agent": "SoccerPredictionResearch/1.0 MOM-label-audit",
    "Referer": "https://www.sofascore.com/",
}
DEFAULT_MAX_EVENT_DELTA_HOURS = 6.0


def _norm_team(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    text = text.casefold()
    # Remove only presentation punctuation/spacing. Do not use fuzzy similarity
    # here: a wrong event label is worse than a missing label.
    return re.sub(r"[^a-z0-9]+", "", text)


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
        url = f"{SOFASCORE_BASE}/unique-tournament/{int(tournament_id)}/season/{int(season_id)}/events/last/{page}"
        response = fetcher.get(
            "sofascore_tournament_season_events",
            url,
            headers=SOFASCORE_HEADERS,
        )
        retrieval_times.append(response.metadata.retrieved_at)
        payload = _parse_json(response.body)
        page_events = payload.get("events")
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


def collect_sofascore_mom_labels_tournament_season(
    fixtures: pd.DataFrame,
    *,
    tournament_id: int,
    season_id: int,
    cache_dir: str = "cache/external",
    max_event_delta_hours: float = DEFAULT_MAX_EVENT_DELTA_HOURS,
    retries: int = 3,
    max_event_pages: int = 200,
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
    events, _ = fetch_tournament_season_events(
        tournament_id,
        season_id,
        fetcher=fetcher,
        max_pages=max_event_pages,
    )

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
    for date_utc in dates:
        events, retrieved_at = fetch_scheduled_events_for_date(date_utc, fetcher=fetcher)
        for event in events:
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
