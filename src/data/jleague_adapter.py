from __future__ import annotations

"""J.League historical adapter with official Data Site fallback."""

import hashlib
import re
import unicodedata
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
FRAME = {"J1": 1, "J2": 2, "J3": 3}
HEADERS = {"User-Agent": "SoccerPredictionResearch/1.0", "Accept-Language": "ja,en;q=0.8"}


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = []

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            text = " ".join(data.split())
            if text:
                self.cell.append(text)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.cell is not None and self.row is not None:
            self.row.append(" ".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.row:
                self.rows.append(self.row)
            self.row = None


def _norm_comp(value: object) -> str | None:
    text = unicodedata.normalize("NFKC", str(value)).strip().upper()
    compact = re.sub(r"[\s._\-]+", "", text)
    aliases = {
        "J1": "J1", "J1LEAGUE": "J1", "JLEAGUEDIVISION1": "J1", "JLEAGUE1": "J1",
        "Jリーグディビジョン1": "J1", "Jリーグ1": "J1",
        "J2": "J2", "J2LEAGUE": "J2", "JLEAGUEDIVISION2": "J2", "JLEAGUE2": "J2",
        "Jリーグディビジョン2": "J2", "Jリーグ2": "J2",
        "J3": "J3", "J3LEAGUE": "J3", "JLEAGUEDIVISION3": "J3", "JLEAGUE3": "J3",
        "Jリーグディビジョン3": "J3", "Jリーグ3": "J3",
    }
    if compact in aliases:
        return aliases[compact]
    for key in ("J3", "J2", "J1"):
        if key in compact or f"JLEAGUE{key[1]}" in compact:
            return key
    return None


def _pick(df: pd.DataFrame, *names: str) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _parse_score(text: str) -> tuple[int | None, int | None]:
    m = re.search(r"(\d+)\s*-\s*(\d+)", str(text))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _load_primary(cache_dir: str, start_year: int, end_year: int) -> pd.DataFrame:
    cache = Path(cache_dir) / "JPN.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        raw = cache.read_bytes()
    else:
        r = requests.get(URL, timeout=60, headers=HEADERS)
        r.raise_for_status()
        raw = r.content
        cache.write_bytes(raw)
    df = pd.read_csv(BytesIO(raw))
    lc, dc = _pick(df, "League", "league"), _pick(df, "Date", "date")
    hc, ac = _pick(df, "Home", "HomeTeam"), _pick(df, "Away", "AwayTeam")
    hgc, agc = _pick(df, "HG", "FTHG"), _pick(df, "AG", "FTAG")
    rc = _pick(df, "Res", "FTR")
    required = {"League": lc, "Date": dc, "Home": hc, "Away": ac, "HG": hgc, "AG": agc, "Res": rc}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"JPN.csv missing columns: {missing}; columns={list(df.columns)}")
    out = pd.DataFrame({
        "competition": df[lc].map(_norm_comp),
        "source_event_date": df[dc].astype("string").str.strip(),
        "home_team": df[hc].astype("string").str.strip(),
        "away_team": df[ac].astype("string").str.strip(),
        "home_goals": pd.to_numeric(df[hgc], errors="coerce"),
        "away_goals": pd.to_numeric(df[agc], errors="coerce"),
        "result": df[rc].astype("string").str.strip(),
    })
    out["date_local"] = pd.to_datetime(out.source_event_date, dayfirst=True, errors="coerce")
    out["kickoff_utc"] = out.date_local.dt.tz_localize("Asia/Tokyo", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    out = out[out.competition.isin(TARGETS) & out.kickoff_utc.notna()].copy()
    out["season_start"] = out.kickoff_utc.dt.year
    out = out[out.season_start.between(start_year, end_year)]
    out["season"] = out.season_start.astype(int).astype(str)
    out["kickoff_time_available"] = False
    out["event_time_precision"] = "DATE_ONLY"
    out["source_name"] = "Football-Data.co.uk:JPN.csv"
    out["source_available_at_utc"] = pd.NaT
    out["retrieved_at_utc"] = datetime.now(timezone.utc)
    out["raw_snapshot_id"] = hashlib.sha256(raw).hexdigest()
    out["source_record_id"] = out.index.astype(str)
    out["match_id"] = [f"jpn:{c}:{y}:{i}" for c, y, i in zip(out.competition, out.season_start, out.source_record_id)]
    return out.drop_duplicates(["competition", "season", "kickoff_utc", "home_team", "away_team"])


def _official_season(comp: str, year: int, cache_dir: str) -> pd.DataFrame:
    # The public search page works reliably with competition_years alone and
    # returns the season's competitions. Filter the returned table locally.
    query_sets = [
        {"competition_years": year, "home_away_select": 0, "tv_relay_station_name": ""},
        {"competition_years": year, "competition_frame_ids": FRAME[comp], "home_away_select": 0, "tv_relay_station_name": ""},
    ]
    cache_dir_p = Path(cache_dir)
    cache_dir_p.mkdir(parents=True, exist_ok=True)
    raw = None
    chosen_url = None
    for idx, params in enumerate(query_sets):
        url = f"{OFFICIAL_URL}?{urlencode(params)}"
        cache = cache_dir_p / f"official_{comp}_{year}_{idx}.html"
        if cache.exists():
            candidate = cache.read_bytes()
        else:
            r = requests.get(url, timeout=60, headers=HEADERS)
            r.raise_for_status()
            candidate = r.content
            cache.write_bytes(candidate)
        if b"J2" in candidate or b"J3" in candidate or "Ｊ２" in candidate.decode("utf-8", "ignore") or "Ｊ３" in candidate.decode("utf-8", "ignore") or "Ｊ１" in candidate.decode("utf-8", "ignore"):
            raw, chosen_url = candidate, url
            break
        raw, chosen_url = candidate, url
    if raw is None:
        return pd.DataFrame()

    parser = _TableParser()
    parser.feed(raw.decode("utf-8", errors="replace"))
    retrieved = datetime.now(timezone.utc)
    rows: list[dict] = []
    for cells in parser.rows:
        if len(cells) < 8:
            continue
        if unicodedata.normalize("NFKC", cells[0]).strip() != str(year):
            continue
        if _norm_comp(cells[1]) != comp:
            continue
        date_text, time_text = cells[3].strip(), cells[4].strip()
        home, score, away = cells[5].strip(), cells[6].strip(), cells[7].strip()
        hg, ag = _parse_score(score)
        dm = re.search(r"(\d{2})/(\d{2})/(\d{2})", date_text)
        if not home or not away or hg is None or ag is None or not dm:
            continue
        yy, mm, dd = map(int, dm.groups())
        local = pd.Timestamp(year=2000 + yy, month=mm, day=dd)
        precision = "DATE_ONLY"
        tm = re.search(r"(\d{1,2}):(\d{2})", time_text)
        if tm:
            local = local.replace(hour=int(tm.group(1)), minute=int(tm.group(2)))
            precision = "MINUTE"
        kickoff = local.tz_localize("Asia/Tokyo").tz_convert("UTC")
        result = "H" if hg > ag else "A" if ag > hg else "D"
        source_id = hashlib.sha256(f"{comp}|{year}|{date_text}|{time_text}|{home}|{away}".encode()).hexdigest()[:24]
        rows.append({
            "match_id": f"jleague-official:{comp}:{year}:{source_id}", "competition": comp,
            "season": str(year), "season_start": year, "kickoff_utc": kickoff,
            "kickoff_time_available": precision == "MINUTE", "event_time_precision": precision,
            "source_event_date": date_text, "home_team": home, "away_team": away,
            "home_goals": hg, "away_goals": ag, "result": result,
            "source_name": "J.League Data Site", "source_record_id": source_id,
            "source_url": chosen_url, "source_available_at_utc": pd.NaT,
            "retrieved_at_utc": retrieved, "raw_snapshot_id": hashlib.sha256(raw).hexdigest(),
        })
    return pd.DataFrame(rows)


def load_jleague_history(start_year: int = 2010, end_year: int = 2025, cache_dir: str = "data/raw/jleague") -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = _load_primary(cache_dir, start_year, end_year)
    frames = [primary]
    coverage: list[dict] = []
    for comp in TARGETS:
        for year in range(start_year, end_year + 1):
            season = str(year)
            if year < MIN_SEASON[comp]:
                coverage.append({"competition": comp, "season": season, "status": "NOT_APPLICABLE", "rows": 0, "source": "J.League Data Site", "reason": f"{comp} did not exist in requested historical season"})
                continue
            primary_rows = primary[(primary.competition == comp) & (primary.season == season)]
            if not primary_rows.empty:
                coverage.append({"competition": comp, "season": season, "status": "AVAILABLE", "rows": len(primary_rows), "source": "Football-Data.co.uk:JPN.csv", "reason": "Observed rows from primary source"})
                continue
            try:
                official = _official_season(comp, year, cache_dir)
                if not official.empty:
                    frames.append(official)
                    coverage.append({"competition": comp, "season": season, "status": "AVAILABLE", "rows": len(official), "source": "J.League Data Site", "reason": "Observed rows from official J.League schedule/results page"})
                else:
                    coverage.append({"competition": comp, "season": season, "status": "UNAVAILABLE", "rows": 0, "source": "J.League Data Site", "reason": "Official J.League page returned no parseable league rows"})
            except Exception as exc:
                coverage.append({"competition": comp, "season": season, "status": "UNAVAILABLE", "rows": 0, "source": "J.League Data Site", "reason": "Official fallback acquisition failed", "error": str(exc)})
    history = pd.concat([x for x in frames if not x.empty], ignore_index=True) if frames else pd.DataFrame()
    if not history.empty:
        history = history.drop_duplicates(["competition", "season", "kickoff_utc", "home_team", "away_team"], keep="first")
        history = history.sort_values(["competition", "kickoff_utc", "home_team", "away_team", "source_name"], kind="mergesort").reset_index(drop=True)
    return history, pd.DataFrame(coverage)
