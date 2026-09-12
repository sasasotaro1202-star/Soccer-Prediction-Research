from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

LEAGUES = {
    "EPL": "E0", "CHA": "E1", "BL1": "D1", "SA": "I1", "LL": "SP1", "FL1": "F1", "ERE": "N1"
}
BASE = "https://www.football-data.co.uk/mmz4281/{season_folder}/{league}.csv"
COMPETITION_TZ = {
    "EPL": "Europe/London", "CHA": "Europe/London", "BL1": "Europe/Berlin",
    "SA": "Europe/Rome", "LL": "Europe/Madrid", "FL1": "Europe/Paris", "ERE": "Europe/Amsterdam",
}
RAW_STAT_MAP = {
    "home_shots": "HS", "away_shots": "AS",
    "home_shots_on_target": "HST", "away_shots_on_target": "AST",
    "home_corners": "HC", "away_corners": "AC",
    "home_fouls": "HF", "away_fouls": "AF",
    "home_yellow_cards": "HY", "away_yellow_cards": "AY",
    "home_red_cards": "HR", "away_red_cards": "AR",
}


def season_folder(start_year: int) -> str:
    return f"{str(start_year)[-2:]}{str(start_year + 1)[-2:]}"


def _parse_football_data_dates(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return pd.to_datetime(text, format="mixed", dayfirst=True, errors="coerce")


def _parse_kickoff(df: pd.DataFrame, competition: str) -> tuple[pd.Series, pd.Series]:
    local_date = _parse_football_data_dates(df["Date"])
    time_text = df["Time"].astype("string").str.strip() if "Time" in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")
    has_time = time_text.notna() & time_text.ne("") & time_text.ne("nan")
    naive = pd.to_datetime(local_date.dt.strftime("%Y-%m-%d") + " " + time_text, format="%Y-%m-%d %H:%M", errors="coerce")
    tz = ZoneInfo(COMPETITION_TZ[competition])
    kickoff = naive.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    date_only = local_date.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    kickoff = kickoff.where(has_time & kickoff.notna(), date_only)
    precision = pd.Series("DATE_ONLY", index=df.index, dtype="string")
    precision = precision.mask(has_time & kickoff.notna(), "MINUTE")
    return kickoff, precision


def load_season(competition: str, start_year: int, cache_dir: str = "data/raw") -> pd.DataFrame:
    if competition not in LEAGUES:
        raise ValueError(f"No Football-Data.co.uk mapping for {competition}")
    league = LEAGUES[competition]
    folder = season_folder(start_year)
    url = BASE.format(season_folder=folder, league=league)
    path = Path(cache_dir) / f"{competition}_{start_year}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raw = path.read_bytes()
    else:
        r = requests.get(url, timeout=30, headers={"User-Agent": "SoccerPredictionResearch/1.0"})
        r.raise_for_status()
        raw = r.content
        path.write_bytes(raw)
    retrieved = datetime.now(timezone.utc)
    df = pd.read_csv(BytesIO(raw))
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{url}: missing columns {sorted(missing)}")
    kickoff, precision = _parse_kickoff(df, competition)
    source_event_date = _parse_football_data_dates(df["Date"]).dt.date.astype("string")
    out = pd.DataFrame({
        "match_id": [f"fd:{competition}:{start_year}:{i}" for i in df.index],
        "competition": competition,
        "season": f"{start_year}/{str(start_year + 1)[-2:]}",
        "season_start": start_year,
        "kickoff_utc": kickoff,
        "kickoff_time_available": precision.eq("MINUTE"),
        "event_time_precision": precision,
        "source_event_date": source_event_date,
        "home_team": df["HomeTeam"].astype(str).str.strip(),
        "away_team": df["AwayTeam"].astype(str).str.strip(),
        "home_goals": pd.to_numeric(df["FTHG"], errors="coerce"),
        "away_goals": pd.to_numeric(df["FTAG"], errors="coerce"),
        "result": df["FTR"].astype(str).str.strip(),
        "source_name": "Football-Data.co.uk",
        "source_record_id": [str(i) for i in df.index],
        "source_available_at_utc": pd.NaT,
        "retrieved_at_utc": retrieved,
    })
    for target, source in RAW_STAT_MAP.items():
        if source in df.columns:
            out[target] = pd.to_numeric(df[source], errors="coerce")
        else:
            out[target] = np.nan
    out["raw_snapshot_id"] = hashlib.sha256(raw).hexdigest()
    return out.dropna(subset=["kickoff_utc", "home_goals", "away_goals"]).reset_index(drop=True)


def _load_one(args):
    comp, year, cache_dir = args
    try:
        d = load_season(comp, year, cache_dir)
        return comp, year, d, {"competition": comp, "season": year, "status": "AVAILABLE", "rows": len(d)}
    except Exception as e:
        return comp, year, None, {"competition": comp, "season": year, "status": "UNAVAILABLE", "rows": 0, "error": str(e)}


def load_available_history(start_year: int = 2010, end_year: int = 2025, max_workers: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Acquire independent competition-season files concurrently.

    Each file remains an immutable raw snapshot; concurrency changes only wall-clock
    time, not the source bytes, parsing rules, or feature values.
    """
    tasks = [(comp, year, "data/raw") for comp in LEAGUES for year in range(start_year, end_year + 1)]
    results = []
    with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), len(tasks)))) as pool:
        futures = [pool.submit(_load_one, t) for t in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda x: (list(LEAGUES).index(x[0]), x[1]))
    frames = [r[2] for r in results if r[2] is not None]
    coverage = [r[3] for r in results]
    history = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not history.empty:
        history = history.sort_values(["competition", "kickoff_utc", "home_team", "away_team"], kind="mergesort").reset_index(drop=True)
    return history, pd.DataFrame(coverage)
