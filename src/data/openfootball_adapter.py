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
}
SEASON_START = {"UCL": 2010, "UEL": 2010, "DFBP": 2010, "CAR": 2010}
DATE_RE = re.compile(r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?.*$")
MATCH_RE = re.compile(r"^\s*(?:(\d{1,2}:\d{2})\s+)?(.+?)\s+v\s+(.+?)\s+(\d+)\s*-\s*(\d+)(.*)$")
SCORE_PAIR_RE = re.compile(r"(\d+)\s*-\s*(\d+)")
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
    """Return regulation/a.e.t. goals for a 1X2 target.

    OpenFootball may encode knockout matches as e.g. ``4-3 pen. (1-1, 0-1)``.
    The leading score is the shootout score, while the first score in the
    parenthesized sequence is the regulation score. For 1X2 training the
    shootout must therefore be discarded. If an explicit a.e.t. score exists,
    use it as the final football score before treating a penalty shootout as a
    draw.
    """
    text = suffix.strip()
    if re.search(r"\bpen\.?\b", text, flags=re.IGNORECASE):
        # Prefer an explicit a.e.t. score when present.
        aet = re.search(r"(\d+)\s*-\s*(\d+)\s+a\.e\.t\.?", text, flags=re.IGNORECASE)
        if aet:
            hg, ag = int(aet.group(1)), int(aet.group(2))
            return hg, ag, "H" if hg > ag else "A" if ag > hg else "D"

        # Otherwise the parenthesized sequence normally contains regulation
        # and, when present, extra-time intermediate scores. The first pair is
        # the regulation score and is the correct 1X2 target.
        paren = re.search(r"\(([^)]*)\)", text)
        if paren:
            pair = SCORE_PAIR_RE.search(paren.group(1))
            if pair:
                hg, ag = int(pair.group(1)), int(pair.group(2))
                return hg, ag, "H" if hg > ag else "A" if ag > hg else "D"
        return final_h, final_a, "D"
    return final_h, final_a, "H" if final_h > final_a else "A" if final_a > final_h else "D"


def parse_football_txt(text: str, competition: str, season_start: int, source_url: str, raw: bytes) -> pd.DataFrame:
    rows: list[dict] = []
    current_date = pd.NaT
    current_year: int | None = None
    previous_month: int | None = None
    retrieved = datetime.now(timezone.utc)
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
                kickoff = kickoff + pd.to_timedelta(hh, unit="h") + pd.to_timedelta(mm, unit="m")
            final_h, final_a = int(m.group(4)), int(m.group(5))
            hg, ag, result = _outcome(final_h, final_a, m.group(6))
        except (TypeError, ValueError):
            continue
        sid = hashlib.sha1(f"{competition}|{season_start}|{line_no}|{home}|{away}|{kickoff.isoformat()}".encode()).hexdigest()[:16]
        rows.append({
            "match_id": f"of:{competition}:{season_start}:{sid}", "competition": competition,
            "season": f"{season_start}/{str(season_start + 1)[-2:]}", "season_start": season_start,
            "kickoff_utc": kickoff, "kickoff_time_available": bool(time_text),
            "event_time_precision": "MINUTE" if time_text else "DATE_ONLY",
            "home_team": home, "away_team": away, "home_goals": hg, "away_goals": ag,
            "result": result, "source_name": "openfootball", "source_record_id": sid,
            "source_url": source_url, "source_available_at_utc": pd.NaT,
            "retrieved_at_utc": retrieved, "raw_snapshot_id": hashlib.sha256(raw).hexdigest(),
        })
    return pd.DataFrame(rows).drop_duplicates(subset=["competition", "season_start", "kickoff_utc", "home_team", "away_team"]) if rows else pd.DataFrame()


def _expected_season_bounds(start_year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    # Allow early qualifiers and late finals, but reject impossible dates that
    # can arise from stale/corrupted cached OpenFootball files.
    lower = pd.Timestamp(datetime(start_year, 6, 1, tzinfo=timezone.utc))
    upper = pd.Timestamp(datetime(start_year + 1, 8, 31, 23, 59, 59, tzinfo=timezone.utc))
    return lower, upper


def _season_dates_valid(frame: pd.DataFrame, start_year: int) -> bool:
    if frame.empty:
        return True
    kickoff = pd.to_datetime(frame["kickoff_utc"], utc=True, errors="coerce")
    if kickoff.isna().any():
        return False
    lower, upper = _expected_season_bounds(start_year)
    return bool((kickoff >= lower).all() and (kickoff <= upper).all())


def load_openfootball_season(competition: str, start_year: int, cache_dir: str = "data/raw/openfootball") -> pd.DataFrame:
    if competition not in BASE_URLS or start_year < SEASON_START[competition]:
        return pd.DataFrame()
    season = _season_folder(start_year)
    url = BASE_URLS[competition].format(season=season)
    cache = Path(cache_dir) / f"{competition}_{season}.txt"
    cache.parent.mkdir(parents=True, exist_ok=True)

    if cache.exists() and cache.stat().st_size > 0:
        raw = cache.read_bytes()
        cached = parse_football_txt(
            raw.decode("utf-8", errors="replace"),
            competition,
            start_year,
            url,
            raw,
        )
        if _season_dates_valid(cached, start_year):
            return cached

    # Stale/corrupt cached content is never used silently. Re-fetch the exact
    # public source and fail closed when the refreshed content is still invalid.
    response = requests.get(url, timeout=45, headers={"User-Agent": "SoccerPredictionResearch/1.0"})
    response.raise_for_status()
    raw = response.content
    parsed = parse_football_txt(
        raw.decode("utf-8", errors="replace"),
        competition,
        start_year,
        url,
        raw,
    )
    if not _season_dates_valid(parsed, start_year):
        lower, upper = _expected_season_bounds(start_year)
        raise ValueError(
            f"OpenFootball {competition} {season} contains kickoff dates outside "
            f"expected season bounds {lower.isoformat()}..{upper.isoformat()}"
        )
    cache.write_bytes(raw)
    return parsed


def load_openfootball_history(start_year: int = 2010, end_year: int = 2025, max_workers: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    tasks = [(c, y) for c in BASE_URLS for y in range(start_year, end_year + 1)]
    def one(c: str, y: int):
        try:
            d = load_openfootball_season(c, y)
            return c, y, d, {"competition": c, "season": f"{y}/{str(y + 1)[-2:]}", "status": "AVAILABLE" if not d.empty else "UNAVAILABLE", "rows": len(d), "source": "openfootball"}
        except Exception as exc:
            return c, y, pd.DataFrame(), {"competition": c, "season": f"{y}/{str(y + 1)[-2:]}", "status": "UNAVAILABLE", "rows": 0, "source": "openfootball", "error": str(exc)}
    results = []
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(tasks)))) as pool:
        futures = [pool.submit(one, c, y) for c, y in tasks]
        for f in as_completed(futures): results.append(f.result())
    results.sort(key=lambda x: (x[0], x[1]))
    frames = [x[2] for x in results if not x[2].empty]
    history = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not history.empty:
        history = history.sort_values(["competition", "kickoff_utc", "home_team", "away_team"], kind="mergesort").reset_index(drop=True)
    return history, pd.DataFrame([x[3] for x in results])
