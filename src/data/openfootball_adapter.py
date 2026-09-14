from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

import pandas as pd

MATCH_RE = re.compile(r"^(?:\s*(\d{1,2}:\d{2})\s+)?(.+?)\s+-\s+(.+?)\s+(\d+)\s*-\s*(\d+)\s*(?:\(([^)]*)\))?\s*$")


def _outcome(home_goals: int, away_goals: int, marker: str | None = None):
    marker = (marker or "").strip().lower()
    # Penalty shootouts do not change the 90-minute 1X2 target.
    if home_goals > away_goals:
        return home_goals, away_goals, "H"
    if home_goals < away_goals:
        return home_goals, away_goals, "A"
    return home_goals, away_goals, "D"


def parse_football_txt(text: str, competition: str, season_start: int) -> pd.DataFrame:
    """Parse openfootball match text without inventing publication timestamps."""
    rows = []
    current_date = pd.NaT
    for line_no, line in enumerate(str(text).splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            parsed_date = pd.to_datetime(line, errors="coerce", dayfirst=True)
        except (TypeError, ValueError):
            parsed_date = pd.NaT
        if pd.notna(parsed_date) and not MATCH_RE.match(line):
            current_date = parsed_date
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
            "match_id": sid,
            "competition": competition,
            "season_start": season_start,
            "kickoff_utc": kickoff.tz_localize(timezone.utc) if kickoff.tzinfo is None else kickoff.tz_convert(timezone.utc),
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
            "result": result,
            "source_name": "openfootball",
            "source_available_at_utc": pd.NaT,
            "pit_evidence_status": "UNVERIFIABLE",
            "pit_evidence_reason": "no_publication_time_evidence",
        })
    return pd.DataFrame(rows)
