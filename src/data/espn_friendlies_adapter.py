from __future__ import annotations

"""Conservative ESPN Club Friendlies adapter.

The ESPN endpoint is treated as an acquisition source only. Retrieval time is
never promoted to publication time, and rows are emitted only when the ESPN
payload contains a parseable fixture/result.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/club.friendly/scoreboard"


def _date_range(start_year: int, end_year: int) -> list[str]:
    return [f"{y}{m:02d}{d:02d}" for y in range(start_year, end_year + 1) for m, d in ((1, 1),)]


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
        if not competitions:
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
        hg, ag = _score(comp)
        status = ((comp.get("status") or {}).get("type") or {}).get("name", "")
        result = pd.NA
        if hg is not None and ag is not None and status in {"STATUS_FINAL", "STATUS_FINAL_OT", "STATUS_FINAL_PEN"}:
            if status == "STATUS_FINAL_PEN":
                result = "D"
            else:
                result = "H" if hg > ag else "A" if ag > hg else "D"
        rows.append({
            "match_id": f"espn:FRI:{event.get('id')}",
            "competition": "FRI",
            "season": str(kickoff.year),
            "season_start": int(kickoff.year),
            "kickoff_utc": kickoff,
            "kickoff_time_available": True,
            "event_time_precision": "MINUTE",
            "home_team": (home.get("team") or {}).get("displayName") or (home.get("team") or {}).get("name"),
            "away_team": (away.get("team") or {}).get("displayName") or (away.get("team") or {}).get("name"),
            "home_goals": hg,
            "away_goals": ag,
            "result": result,
            "source_name": "ESPN:club.friendly",
            "source_record_id": str(event.get("id")),
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
    for year in range(start_year, end_year + 1):
        # ESPN's scoreboard supports date-scoped queries. We intentionally query
        # the full year in monthly chunks to avoid oversized responses.
        year_rows: list[dict] = []
        for month in range(1, 13):
            start = f"{year}{month:02d}01"
            if month == 12:
                end = f"{year}1231"
            else:
                next_month = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(1)
                end = next_month.strftime("%Y%m%d")
            path = cache / f"{year}_{month:02d}.json"
            try:
                if path.exists():
                    raw = path.read_bytes()
                else:
                    r = requests.get(f"{BASE}?dates={start}-{end}", timeout=60, headers={"User-Agent": "SoccerPredictionResearch/1.0"})
                    r.raise_for_status()
                    raw = r.content
                    path.write_bytes(raw)
                payload = json.loads(raw.decode("utf-8"))
                retrieved = datetime.now(timezone.utc)
                snapshot = hashlib.sha256(raw).hexdigest()
                year_rows.extend(_event_rows(payload, snapshot, retrieved))
            except Exception as exc:
                coverage.append({"competition": "FRI", "season": str(year), "source": "ESPN:club.friendly", "status": "PARSE_ERROR", "rows": 0, "error": str(exc)})
        if year_rows:
            rows.extend(year_rows)
            coverage.append({"competition": "FRI", "season": str(year), "source": "ESPN:club.friendly", "status": "AVAILABLE", "rows": len(year_rows)})
        elif not any(x["season"] == str(year) and x["status"] == "PARSE_ERROR" for x in coverage):
            coverage.append({"competition": "FRI", "season": str(year), "source": "ESPN:club.friendly", "status": "UNAVAILABLE", "rows": 0, "reason": "No observed parseable rows from ESPN scoreboard"})
    history = pd.DataFrame(rows)
    if history.empty:
        return history, pd.DataFrame(coverage)
    history = history.drop_duplicates(subset=["competition", "season", "kickoff_utc", "home_team", "away_team"], keep="first")
    history = history.sort_values(["season_start", "kickoff_utc", "home_team", "away_team"], kind="mergesort").reset_index(drop=True)
    return history, pd.DataFrame(coverage)
