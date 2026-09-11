from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

LEAGUES = {
    "EPL": "E0", "CHA": "E1", "BL1": "D1", "SA": "I1", "LL": "SP1", "FL1": "F1", "ERE": "N1"
}
BASE = "https://www.football-data.co.uk/mmz4281/{season_folder}/{league}.csv"


def season_folder(start_year: int) -> str:
    return f"{str(start_year)[-2:]}{str(start_year + 1)[-2:]}"


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
    date = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce", utc=True)
    # Football-Data rows represent completed historical matches. We intentionally
    # do not pretend retrieved_at is source_available_at; availability is unknown.
    out = pd.DataFrame({
        "match_id": [f"fd:{competition}:{start_year}:{i}" for i in df.index],
        "competition": competition,
        "season": f"{start_year}/{str(start_year + 1)[-2:]}",
        "kickoff_utc": date,
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
    # The raw snapshot is reproducible even when source publication time is absent.
    digest = hashlib.sha256(raw).hexdigest()
    out["raw_snapshot_id"] = digest
    return out.dropna(subset=["kickoff_utc", "home_goals", "away_goals"]).reset_index(drop=True)


def load_available_history(start_year: int = 2010, end_year: int = 2025) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, coverage = [], []
    for comp in LEAGUES:
        for year in range(start_year, end_year + 1):
            try:
                d = load_season(comp, year)
                frames.append(d)
                coverage.append({"competition": comp, "season": year, "status": "AVAILABLE", "rows": len(d)})
            except Exception as e:
                coverage.append({"competition": comp, "season": year, "status": "UNAVAILABLE", "rows": 0, "error": str(e)})
    history = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return history, pd.DataFrame(coverage)
