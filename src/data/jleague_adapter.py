from __future__ import annotations

"""Conservative adapter for Football-Data.co.uk's Japan historical CSV.

Competition is taken from the source League column. Requested season cells are
always emitted into coverage so absence is explicit and never confused with a
zero or an unqueried cell.
"""

import hashlib
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

URL = "https://football-data.co.uk/new/JPN.csv"
TARGETS = {"J1": "J1", "J2": "J2", "J3": "J3"}
MIN_SEASON = {"J1": 2010, "J2": 2010, "J3": 2014}


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


def load_jleague_history(start_year: int = 2010, end_year: int = 2025, cache_dir: str = "data/raw/jleague") -> tuple[pd.DataFrame, pd.DataFrame]:
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
    out = out.sort_values(["competition", "kickoff_utc", "home_team", "away_team"], kind="mergesort").reset_index(drop=True)
    coverage_rows = []
    for comp in ("J1", "J2", "J3"):
        for season_start in range(start_year, end_year + 1):
            season = str(season_start)
            g = out[(out["competition"] == comp) & (out["season"] == season)]
            if season_start < MIN_SEASON[comp]:
                status, reason = "NOT_APPLICABLE", f"{comp} did not exist in requested historical season {season}"
            elif len(g):
                status, reason = "AVAILABLE", "Observed rows from JPN.csv"
            else:
                status, reason = "UNAVAILABLE", "No observed rows in JPN.csv for requested season"
            coverage_rows.append({"competition": comp, "season": season, "status": status, "rows": len(g), "source": "Football-Data.co.uk:JPN.csv", "reason": reason})
    return out, pd.DataFrame(coverage_rows)
