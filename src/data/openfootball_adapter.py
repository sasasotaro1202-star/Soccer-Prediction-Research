from __future__ import annotations

"""Conservative adapter for public-domain Football.TXT historical results."""

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URLS = {
    "UCL": "https://raw.githubusercontent.com/openfootball/champions-league/master/{season}/cl.txt",
    "UEL": "https://raw.githubusercontent.com/openfootball/champions-league/master/{season}/el.txt",
    "DFBP": "https://raw.githubusercontent.com/openfootball/deutschland/master/{season}/cup.txt",
    "CAR": "https://raw.githubusercontent.com/openfootball/england/master/{season}/eflcup.txt",
}
SEASON_START = {"UCL": 2011, "UEL": 2011, "DFBP": 2010, "CAR": 2010}
DATE_RE = re.compile(r"^\s{2}(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
MATCH_RE = re.compile(r"^\s{4}(?:(\d{1,2}:\d{2})\s+)?(.+?)\s+v\s+(.+?)\s+(\d+)\s*-\s*(\d+)(.*)$")
MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def _season_folder(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _parse_date(line: str, season_start: int, current_year: int | None, previous_month: int | None) -> tuple[pd.Timestamp, int | None, int | None]:
    m = DATE_RE.match(line)
    if not m:
        return pd.NaT, current_year, previous_month
    month = MONTHS[m.group(1)]
    if m.group(3):
        year = int(m.group(3))
    elif current_year is None:
        year = season_start if month >= 7 else season_start + 1
    elif previous_month is not None and month < previous_month:
        year = current_year + 1
    else:
        year = current_year
    try:
        ts = pd.Timestamp(datetime(year, month, int(m.group(2)), tzinfo=timezone.utc))
    except ValueError:
        return pd.NaT, year, month
    return ts, year, month


def _outcome(final_h: int, final_a: int, suffix: str) -> tuple[int, int, str]:
    text = suffix.strip()
    # Football.TXT can encode shootouts as: "1-4 pen. 0-1 a.e.t.".
    # The competition target remains a draw because regulation was level.
    if re.search(r"\bpen\.\b", text):
        reg = re.search(r"(\d+)\s*-\s*(\d+)\s+a\.e\.t\.", text)
        if reg:
            return int(reg.group(1)), int(reg.group(2)), "D"
        return final_h, final_a, "D"
    return final_h, final_a, "H" if final_h > final_a else "A" if final_a > final_h else "D"


def parse_football_txt(text: str, competition: str, season_start: int, source_url: str, raw: bytes) -> pd.DataFrame:
    rows: list[dict] = []
    current_date = pd.NaT
    current_year: int | None = None
    previous_month: int | None = None
    for line_no, line in enumerate(text.splitlines(), 1):
        if DATE_RE.match(line):
            current_date, current_year, previous_month = _parse_date(line, season_start, current_year, previous_month)
            continue
        m = MATCH_RE.match(line)
        if not m or pd.isna(current_date):
            continue
        try:
            time_text, home, away = m.group(1), m.group(2).strip(), m.group(3).strip()
            kickoff = current_date
            if time_text:
                hh, mm = map(int, time_text.split(":"))
                kickoff = kickoff + pd.Timedelta(hours=hh, minutes=mm)
            final_h, final_a = int(m.group(4)), int(m.group(5))
            hg, ag, result = _outcome(final_h, final_a, m.group(6))
        except (TypeError, ValueError):
            continue
        sid = hashlib.sha1(f"{competition}|{season_start}|{line_no}|{home}|{away}|{kickoff.isoformat()}".encode()).hexdigest()[:16]
        rows.append({
            "match_id": f"of:{competition}:{season_start}:{sid}",
            "competition": competition,
            "season": f"{season_start}/{str(season_start + 1)[-2:]}",
            "season_start": season_start,
            "kickoff_utc": kickoff,
            "kickoff_time_available": bool(time_text),
            "event_time_precision": "MINUTE" if time_text else "DATE_ONLY",
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
            "result": result,
            "source_name": "openfootball",
            "source_record_id": sid,
            "source_url": source_url,
            "source_available_at_utc": pd.NaT,
            "retrieved_at_utc": datetime.now(timezone.utc),
            "raw_snapshot_id": hashlib.sha256(raw).hexdigest(),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["competition", "season_start", "kickoff_utc", "home_team", "away_team"])


def load_openfootball_season(competition: str, start_year: int, cache_dir: str = "data/raw/openfootball") -> pd.DataFrame:
    if competition not in BASE_URLS:
        raise ValueError(f"No openfootball mapping for {competition}")
    if start_year < SEASON_START[competition]:
        return pd.DataFrame()
    season = _season_folder(start_year)
    url = BASE_URLS[competition].format(season=season)
    cache = Path(cache_dir) / f"{competition}_{season}.txt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        raw = cache.read_bytes()
    else:
        r = requests.get(url, timeout=30, headers={"User-Agent": "SoccerPredictionResearch/1.0"})
        r.raise_for_status()
        raw = r.content
        cache.write_bytes(raw)
    return parse_football_txt(raw.decode("utf-8", errors="replace"), competition, start_year, url, raw)


def load_openfootball_history(start_year: int = 2010, end_year: int = 2025, max_workers: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    tasks = [(c, y) for c in BASE_URLS for y in range(start_year, end_year + 1)]
    def one(c: str, y: int):
        try:
            d = load_openfootball_season(c, y)
            return c, y, d, {"competition": c, "season": y, "status": "AVAILABLE" if not d.empty else "UNAVAILABLE", "rows": len(d), "source": "openfootball"}
        except Exception as exc:
            return c, y, pd.DataFrame(), {"competition": c, "season": y, "status": "UNAVAILABLE", "rows": 0, "source": "openfootball", "error": str(exc)}
    results = []
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(tasks)))) as pool:
        futures = [pool.submit(one, c, y) for c, y in tasks]
        for f in as_completed(futures):
            results.append(f.result())
    results.sort(key=lambda x: (x[0], x[1]))
    frames = [x[2] for x in results if not x[2].empty]
    history = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not history.empty:
        history = history.sort_values(["competition", "kickoff_utc", "home_team", "away_team"], kind="mergesort").reset_index(drop=True)
    return history, pd.DataFrame([x[3] for x in results])
