from __future__ import annotations

"""Conservative ESPN Club Friendlies adapter.

ESPN's public site API is undocumented, so every response is cached and hashed.
Retrieval time is never promoted to publication time. Friendlies are assigned
to European-style seasons (Jul-Jun) for the audit; missing historical ESPN data
remains explicitly unavailable rather than being synthesized.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/club.friendly/scoreboard"
HEADERS = {"User-Agent": "python-requests/SoccerPredictionResearch"}


def _season_start(ts: pd.Timestamp) -> int:
    return int(ts.year if ts.month >= 7 else ts.year - 1)


def _score(comp: dict) -> tuple[float | None, float | None]:
    competitors = comp.get("competitors") or []
    home = next((x for x in competitors if x.get("homeAway") == "home"), None)
    away = next((x for x in competitors if x.get("homeAway") == "away"), None)
    try:
        return float(home["score"]), float(away["score"])
    except (TypeError, KeyError):
        return None, None


def _event_rows(payload: dict, snapshot_id: str, retrieved: datetime) -> list[dict]:
    rows: list[dict] = []
    for event in payload.get("events") or []:
        competitions = event.get("competitions") or []
        if not competitions or not event.get("id"):
            continue
        comp = competitions[0]
        competitors = comp.get("competitors") or []
        home = next((x for x in competitors if x.get("homeAway") == "home"), None)
        away = next((x for x in competitors if x.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        kickoff = pd.to_datetime(event.get("date"), utc=True, errors="coerce")
        if pd.isna(kickoff):
            continue
        home_team = home.get("team") or {}
        away_team = away.get("team") or {}
        home_name = home_team.get("displayName") or home_team.get("name")
        away_name = away_team.get("displayName") or away_team.get("name")
        if not home_name or not away_name:
            continue
        hg, ag = _score(comp)
        status = ((comp.get("status") or {}).get("type") or {}).get("name", "")
        result = pd.NA
        if hg is not None and ag is not None and status in {"STATUS_FINAL", "STATUS_FINAL_OT", "STATUS_FINAL_PEN"}:
            if status == "STATUS_FINAL_PEN":
                result = "D"
            else:
                result = "H" if hg > ag else "A" if ag > hg else "D"
        ss = _season_start(kickoff)
        rows.append({
            "match_id": f"espn:FRI:{event['id']}",
            "competition": "FRI",
            "season": f"{ss}/{str(ss + 1)[-2:]}",
            "season_start": ss,
            "kickoff_utc": kickoff,
            "kickoff_time_available": True,
            "event_time_precision": "MINUTE",
            "home_team": str(home_name).strip(),
            "away_team": str(away_name).strip(),
            "home_team_id": str(home_team.get("id")) if home_team.get("id") is not None else pd.NA,
            "away_team_id": str(away_team.get("id")) if away_team.get("id") is not None else pd.NA,
            "home_goals": hg,
            "away_goals": ag,
            "result": result,
            "source_name": "ESPN:club.friendly",
            "source_record_id": str(event["id"]),
            "source_url": BASE,
            "source_available_at_utc": pd.NaT,
            "retrieved_at_utc": retrieved,
            "raw_snapshot_id": snapshot_id,
        })
    return rows


def load_friendlies_history(start_year: int = 2010, end_year: int = 2025, cache_dir: str = "data/raw/espn_friendlies") -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    coverage: list[dict] = []
    for calendar_year in range(start_year, end_year + 2):
        for month in range(1, 13):
            start = f"{calendar_year}{month:02d}01"
            end = (pd.Timestamp(calendar_year, month, 1) + pd.offsets.MonthEnd(1)).strftime("%Y%m%d")
            path = cache / f"{calendar_year}_{month:02d}.json"
            try:
                if path.exists():
                    raw = path.read_bytes()
                else:
                    r = requests.get(BASE, params={"dates": f"{start}-{end}", "limit": 1000}, timeout=60, headers=HEADERS)
                    r.raise_for_status()
                    raw = r.content
                    path.write_bytes(raw)
                payload = json.loads(raw.decode("utf-8"))
                snapshot = hashlib.sha256(raw).hexdigest()
                rows.extend(_event_rows(payload, snapshot, datetime.now(timezone.utc)))
            except Exception as exc:
                coverage.append({"competition": "FRI", "calendar_year": calendar_year, "month": month, "status": "PARSE_ERROR", "rows": 0, "source": "ESPN:club.friendly", "error": str(exc)})
    history = pd.DataFrame(rows)
    if history.empty:
        return history, pd.DataFrame(coverage)
    history = history[(history["season_start"] >= start_year) & (history["season_start"] <= end_year)].copy()
    # ESPN event IDs are authoritative within this source. A canonical-key duplicate
    # is retained only once; same-source duplicates are surfaced separately by the audit.
    history = history.drop_duplicates(subset=["source_record_id"], keep="first")
    history = history.sort_values(["season_start", "kickoff_utc", "home_team", "away_team"], kind="mergesort").reset_index(drop=True)
    season_rows = history.groupby("season", dropna=False).size().to_dict()
    season_status = []
    for ss in range(start_year, end_year + 1):
        season = f"{ss}/{str(ss + 1)[-2:]}"
        season_status.append({"competition": "FRI", "season": season, "status": "AVAILABLE" if season_rows.get(season, 0) else "UNAVAILABLE", "rows": int(season_rows.get(season, 0)), "source": "ESPN:club.friendly", "reason": "Observed ESPN event rows" if season_rows.get(season, 0) else "No observed parseable ESPN event rows"})
    return history, pd.DataFrame(season_status + coverage)
