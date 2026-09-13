from __future__ import annotations

"""Conservative J.League historical adapter.

Primary source: Football-Data.co.uk JPN.csv when it contains the requested
cell. Fallback source: the official J.League Data Site schedule/results pages
for cells missing from JPN.csv, especially J2/J3. The fallback only adds
observed match rows; it never converts absence into zero or fabricates
publication timestamps.
"""

import hashlib
from datetime import datetime, timezone
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

URL = "https://football-data.co.uk/new/JPN.csv"
OFFICIAL_URL = "https://data.j-league.or.jp/SFMS01/search"
TARGETS = {"J1": "J1", "J2": "J2", "J3": "J3"}
MIN_SEASON = {"J1": 2010, "J2": 1999, "J3": 2014}
OFFICIAL_FRAME = {"J1": 1, "J2": 2, "J3": 3}


class _TableParser(HTMLParser):
    """Small stdlib-only HTML table parser; avoids a heavy parser dependency."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            text = " ".join(data.split())
            if text:
                self._cell.append(text)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(" ".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


def _pick(df: pd.DataFrame, *names: str) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _norm_comp(value: object) -> str | None:
    text = str(value).strip().upper()
    aliases = {
        "J1": "J1", "J1 LEAGUE": "J1", "J.LEAGUE DIVISION 1": "J1",
        "J2": "J2", "J2 LEAGUE": "J2", "J.LEAGUE DIVISION 2": "J2",
        "J3": "J3", "J3 LEAGUE": "J3", "J.LEAGUE DIVISION 3": "J3",
    }
    return aliases.get(text)


def _load_primary(cache_dir: str, start_year: int, end_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = Path(cache_dir) / "JPN.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        raw = cache.read_bytes()
    else:
        response = requests.get(URL, timeout=60, headers={"User-Agent": "SoccerPredictionResearch/1.0"})
        response.raise_for_status()
        raw = response.content
        cache.write_bytes(raw)

    retrieved = datetime.now(timezone.utc)
    df = pd.read_csv(BytesIO(raw))
    league_col = _pick(df, "League", "Liga", "league")
    date_col = _pick(df, "Date", "date", "Data")
    home_col = _pick(df, "Home", "HomeTeam", "home_team", "Mandante")
    away_col = _pick(df, "Away", "AwayTeam", "away_team", "Visitante")
    hg_col = _pick(df, "HG", "FTHG", "home_goals", "Gols Mandante")
    ag_col = _pick(df, "AG", "FTAG", "away_goals", "Gols Visitante")
    result_col = _pick(df, "Res", "FTR", "result", "Resultado")
    required = {"League": league_col, "Date": date_col, "Home": home_col, "Away": away_col, "HG": hg_col, "AG": ag_col, "Res": result_col}
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError(f"{URL}: missing required columns {missing}; columns={list(df.columns)}")

    out = pd.DataFrame({
        "competition": df[league_col].map(_norm_comp),
        "source_event_date": df[date_col].astype("string").str.strip(),
        "home_team": df[home_col].astype("string").str.strip(),
        "away_team": df[away_col].astype("string").str.strip(),
        "home_goals": pd.to_numeric(df[hg_col], errors="coerce"),
        "away_goals": pd.to_numeric(df[ag_col], errors="coerce"),
        "result": df[result_col].astype("string").str.strip(),
    })
    out["date_local"] = pd.to_datetime(out["source_event_date"], dayfirst=True, errors="coerce")
    out["kickoff_utc"] = out["date_local"].dt.tz_localize("Asia/Tokyo", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    out = out[(out["competition"].isin(TARGETS)) & out["kickoff_utc"].notna()].copy()
    out["season_start"] = out["kickoff_utc"].dt.year
    out = out[(out["season_start"] >= start_year) & (out["season_start"] <= end_year)]
    out["season"] = out["season_start"].astype(int).astype(str)
    out["kickoff_time_available"] = False
    out["event_time_precision"] = "DATE_ONLY"
    out["source_name"] = "Football-Data.co.uk:JPN.csv"
    out["source_available_at_utc"] = pd.NaT
    out["retrieved_at_utc"] = retrieved
    out["raw_snapshot_id"] = hashlib.sha256(raw).hexdigest()
    out["source_record_id"] = out.index.astype(str)
    out["match_id"] = [f"jpn:{c}:{y}:{i}" for c, y, i in zip(out["competition"], out["season_start"], out["source_record_id"])]
    out = out.drop_duplicates(subset=["competition", "season", "kickoff_utc", "home_team", "away_team"])
    return out.sort_values(["competition", "kickoff_utc", "home_team", "away_team"], kind="mergesort").reset_index(drop=True), pd.DataFrame()


def _parse_score(text: str) -> tuple[int | None, int | None]:
    import re
    m = re.search(r"(\d+)\s*-\s*(\d+)", text)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _official_season(comp: str, year: int, cache_dir: str = "data/raw/jleague") -> pd.DataFrame:
    params = urlencode({
        "competition_years": year,
        "competition_frame_ids": OFFICIAL_FRAME[comp],
        "home_away_select": 0,
        "tv_relay_station_name": "",
    })
    url = f"{OFFICIAL_URL}?{params}"
    cache = Path(cache_dir) / f"official_{comp}_{year}.html"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        raw = cache.read_bytes()
    else:
        response = requests.get(url, timeout=60, headers={"User-Agent": "SoccerPredictionResearch/1.0"})
        response.raise_for_status()
        raw = response.content
        cache.write_bytes(raw)

    parser = _TableParser()
    parser.feed(raw.decode("utf-8", errors="replace"))
    retrieved = datetime.now(timezone.utc)
    rows = []
    for cells in parser.rows:
        if len(cells) < 8 or cells[0] != str(year):
            continue
        league = _norm_comp(cells[1])
        if league != comp:
            continue
        date_text, time_text = cells[3].strip(), cells[4].strip()
        home, score, away = cells[5].strip(), cells[6].strip(), cells[7].strip()
        hg, ag = _parse_score(score)
        if not home or not away or hg is None or ag is None:
            continue
        # Official pages use JST local time. Preserve DATE_ONLY when kickoff is absent.
        date_match = __import__("re").search(r"(\d{2})/(\d{2})/(\d{2})", date_text)
        if not date_match:
            continue
        yy, mm, dd = map(int, date_match.groups())
        local = pd.Timestamp(year=2000 + yy, month=mm, day=dd)
        precision = "DATE_ONLY"
        if __import__("re").fullmatch(r"\d{1,2}:\d{2}", time_text):
            hh, minute = map(int, time_text.split(":"))
            local = local.replace(hour=hh, minute=minute)
            precision = "MINUTE"
        kickoff = local.tz_localize("Asia/Tokyo").tz_convert("UTC")
        result = "H" if hg > ag else "A" if hg < ag else "D"
        source_id = hashlib.sha256(f"{comp}|{year}|{date_text}|{time_text}|{home}|{away}".encode()).hexdigest()[:24]
        rows.append({
            "match_id": f"jleague-official:{comp}:{year}:{source_id}",
            "competition": comp,
            "season": str(year),
            "season_start": year,
            "kickoff_utc": kickoff,
            "kickoff_time_available": precision == "MINUTE",
            "event_time_precision": precision,
            "source_event_date": date_text,
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
            "result": result,
            "source_name": "J.League Data Site",
            "source_record_id": source_id,
            "source_available_at_utc": pd.NaT,
            "retrieved_at_utc": retrieved,
            "raw_snapshot_id": hashlib.sha256(raw).hexdigest(),
        })
    return pd.DataFrame(rows)


def load_jleague_history(start_year: int = 2010, end_year: int = 2025, cache_dir: str = "data/raw/jleague") -> tuple[pd.DataFrame, pd.DataFrame]:
    primary, _ = _load_primary(cache_dir, start_year, end_year)
    frames = [primary]
    coverage_rows = []
    for comp in ("J1", "J2", "J3"):
        for season_start in range(start_year, end_year + 1):
            season = str(season_start)
            if season_start < MIN_SEASON[comp]:
                coverage_rows.append({"competition": comp, "season": season, "status": "NOT_APPLICABLE", "rows": 0, "source": "J.League Data Site", "reason": f"{comp} did not exist in requested historical season {season}"})
                continue
            g = primary[(primary["competition"] == comp) & (primary["season"] == season)]
            if len(g):
                coverage_rows.append({"competition": comp, "season": season, "status": "AVAILABLE", "rows": len(g), "source": "Football-Data.co.uk:JPN.csv", "reason": "Observed rows from primary source"})
                continue
            try:
                official = _official_season(comp, season_start, cache_dir)
                if not official.empty:
                    frames.append(official)
                    coverage_rows.append({"competition": comp, "season": season, "status": "AVAILABLE", "rows": len(official), "source": "J.League Data Site", "reason": "Observed rows from official J.League schedule/results page"})
                else:
                    coverage_rows.append({"competition": comp, "season": season, "status": "UNAVAILABLE", "rows": 0, "source": "J.League Data Site", "reason": "Official J.League page returned no parseable league rows"})
            except Exception as exc:
                coverage_rows.append({"competition": comp, "season": season, "status": "UNAVAILABLE", "rows": 0, "source": "J.League Data Site", "reason": "Official fallback acquisition failed", "error": str(exc)})
    history = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    if not history.empty:
        history = history.drop_duplicates(subset=["competition", "season", "kickoff_utc", "home_team", "away_team"], keep="first")
        history = history.sort_values(["competition", "kickoff_utc", "home_team", "away_team", "source_name"], kind="mergesort").reset_index(drop=True)
    return history, pd.DataFrame(coverage_rows)
