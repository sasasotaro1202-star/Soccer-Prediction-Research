from __future__ import annotations

"""Free public international-match adapter.

This adapter deliberately separates tournament coverage from production eligibility.
The upstream openfootball/internationals dataset is CC0/public-domain style data,
but it does not provide publication timestamps. Therefore every row is marked
PIT_UNKNOWN and the research engine must not use these rows for historical model
training until a publication-time source is proven.
"""

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = "https://raw.githubusercontent.com/openfootball/internationals/master/{directory}/{year}_{filename}.txt"

# Senior competitions that materially improve international-strength context.
# Youth/high-school competitions are intentionally separate because the public
# senior dataset explicitly excludes U23/youth and league-select matches.
TOURNAMENTS = {
    "WORLD_CUP": ("fifa_world_cup", "fifa_world_cup"),
    "WORLD_CUP_QUALI": ("fifa_world_cup_qualification", "fifa_world_cup_qualification"),
    "ASIAN_CUP": ("afc_asian_cup", "afc_asian_cup"),
    "EURO": ("uefa_euro", "uefa_euro"),
    "EURO_QUALI": ("uefa_euro_qualification", "uefa_euro_qualification"),
    "NATIONS_LEAGUE": ("uefa_nations_league", "uefa_nations_league"),
    "INTERNATIONAL_FRIENDLY": ("friendly", "friendly"),
}

DATE_RE = re.compile(r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?.*$")
MATCH_RE = re.compile(r"^\s*(?:\(\d+\)\s*)?(?:(\d{1,2}:\d{2})\s+)?(.+?)\s+(\d+)\s*-\s*(\d+)(.*?)(?:\s+@.*)?$")
MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
HEADERS = {"User-Agent": "SoccerPredictionResearch/1.0"}


def _season_start_year(year: int) -> int:
    return int(year)


def _parse_date(line: str, season_year: int, current_year: int | None, previous_month: int | None):
    m = DATE_RE.match(line)
    if not m:
        return pd.NaT, current_year, previous_month
    month = MONTHS[m.group(1)]
    if m.group(3):
        year = int(m.group(3))
    elif current_year is None:
        year = season_year
    elif previous_month is not None and month < previous_month:
        year = current_year + 1
    else:
        year = current_year
    try:
        ts = pd.Timestamp(datetime(year, month, int(m.group(2)), tzinfo=timezone.utc))
    except ValueError:
        return pd.NaT, year, month
    return ts, year, month


def _target_from_score(home_goals: int, away_goals: int, suffix: str):
    text = str(suffix or "")
    # Shootout-only suffixes mean regulation was a draw. This preserves the
    # same 1X2 target convention as the core model.
    if re.search(r"penalties|on pens|wins on penalties|won on penalties", text, flags=re.I):
        return home_goals, away_goals, "D", True
    # When extra time is explicitly present but no regulation score is given,
    # keep the final score but flag the target as not regulation-verified.
    regulation_verified = not bool(re.search(r"\ba\.e\.t\.?\b|\baet\b", text, flags=re.I))
    result = "H" if home_goals > away_goals else "A" if away_goals > home_goals else "D"
    return home_goals, away_goals, result, regulation_verified


def parse_football_txt(text: str, competition: str, season_year: int, source_url: str, raw: bytes) -> pd.DataFrame:
    rows: list[dict] = []
    current_date = pd.NaT
    current_year = None
    previous_month = None
    retrieved = datetime.now(timezone.utc)
    for line_no, line in enumerate(text.splitlines(), 1):
        if DATE_RE.match(line):
            current_date, current_year, previous_month = _parse_date(line, season_year, current_year, previous_month)
            continue
        if pd.isna(current_date):
            continue
        m = MATCH_RE.match(line)
        if not m:
            continue
        try:
            time_text, home, away = m.group(1), m.group(2).strip(), None
            # MATCH_RE's final suffix is deliberately broad; recover the away
            # team from the portion between the score and the optional venue.
            # The first two numeric groups are always the score.
            hg, ag = int(m.group(3)), int(m.group(4))
            tail = m.group(5).strip()
            # Re-run a stricter split on the full match line to avoid treating
            # a team name containing punctuation as metadata.
            line_core = re.sub(r"\s+@.*$", "", line).strip()
            score_match = re.search(r"\s(\d+)\s*-\s*(\d+)(?:\s|$)", line_core)
            if not score_match:
                continue
            left = line_core[: score_match.start()].strip()
            right = line_core[score_match.end() :].strip()
            if time_text and left.startswith(time_text):
                left = left[len(time_text):].strip()
            left = re.sub(r"^\(\d+\)\s*", "", left).strip()
            # A single run of 2+ spaces is the standard Football.TXT team
            # separator. Fall back to a conservative midpoint split.
            parts = re.split(r"\s{2,}", left, maxsplit=1)
            if len(parts) == 2:
                home, away = parts[0].strip(), parts[1].strip()
            else:
                # Most datasets align the two team names in fixed-width text;
                # use the score position as the final separator.
                words = left.split()
                if len(words) < 2:
                    continue
                midpoint = max(1, len(words) // 2)
                home, away = " ".join(words[:midpoint]), " ".join(words[midpoint:])
            if not home or not away or home.lower() == away.lower():
                continue
            kickoff = current_date
            if time_text:
                hh, mm = map(int, time_text.split(":"))
                kickoff = kickoff + pd.Timedelta(hours=hh, minutes=mm)
            hg, ag, result, regulation_verified = _target_from_score(hg, ag, tail)
        except (TypeError, ValueError):
            continue
        source_id = hashlib.sha1(
            f"{competition}|{season_year}|{line_no}|{home}|{away}|{kickoff.isoformat()}".encode()
        ).hexdigest()[:20]
        rows.append({
            "match_id": f"intl:{competition}:{season_year}:{source_id}",
            "competition": competition,
            "season": str(season_year),
            "season_start": season_year,
            "kickoff_utc": kickoff,
            "kickoff_time_available": bool(time_text),
            "event_time_precision": "MINUTE" if time_text else "DATE_ONLY",
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
            "result": result,
            "regulation_result_verified": regulation_verified,
            "source_name": "openfootball/internationals",
            "source_record_id": source_id,
            "source_url": source_url,
            "source_available_at_utc": pd.NaT,
            "retrieved_at_utc": retrieved,
            "raw_snapshot_id": hashlib.sha256(raw).hexdigest(),
            "pit_status": "PIT_UNKNOWN",
            "production_eligible": False,
        })
    return pd.DataFrame(rows)


def load_international_season(competition: str, season_year: int, cache_dir: str = "data/raw/internationals") -> pd.DataFrame:
    if competition not in TOURNAMENTS:
        return pd.DataFrame()
    directory, filename = TOURNAMENTS[competition]
    url = BASE.format(directory=directory, year=season_year, filename=filename)
    cache = Path(cache_dir) / f"{competition}_{season_year}.txt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        raw = cache.read_bytes()
    else:
        response = requests.get(url, timeout=45, headers=HEADERS)
        if response.status_code == 404:
            return pd.DataFrame()
        response.raise_for_status()
        raw = response.content
        cache.write_bytes(raw)
    return parse_football_txt(raw.decode("utf-8", errors="replace"), competition, season_year, url, raw)


def load_international_history(start_year: int = 2010, end_year: int = 2026, max_workers: int = 8):
    tasks = [(c, y) for c in TOURNAMENTS for y in range(start_year, end_year + 1)]

    def one(c: str, y: int):
        try:
            frame = load_international_season(c, y)
            return c, y, frame, {
                "competition": c,
                "season": str(y),
                "status": "AVAILABLE" if not frame.empty else "UNAVAILABLE",
                "rows": len(frame),
                "source": "openfootball/internationals",
                "pit_status": "PIT_UNKNOWN",
                "production_eligible": False,
            }
        except Exception as exc:
            return c, y, pd.DataFrame(), {
                "competition": c,
                "season": str(y),
                "status": "UNAVAILABLE",
                "rows": 0,
                "source": "openfootball/internationals",
                "pit_status": "PIT_UNKNOWN",
                "production_eligible": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    results = []
    with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), len(tasks)))) as pool:
        futures = [pool.submit(one, c, y) for c, y in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda x: (x[0], x[1]))
    frames = [r[2] for r in results if not r[2].empty]
    history = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not history.empty:
        history = history.drop_duplicates(["competition", "season", "kickoff_utc", "home_team", "away_team"], keep="first")
        history = history.sort_values(["competition", "kickoff_utc", "home_team", "away_team"], kind="mergesort").reset_index(drop=True)
    return history, pd.DataFrame([r[3] for r in results])
