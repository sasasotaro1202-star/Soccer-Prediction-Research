"""Research-only historical player contribution adapter for the Global Football Data Lake.

Player match facts are post-match observations. They are joined to the fixture-level
match_stats.known_at timestamp and therefore enter downstream feature replay only
when that timestamp is <= the prediction cutoff.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.data.competition_sources import TARGET_COMPETITIONS
from src.data.global_datalake_adapter import LEAGUE_TO_COMPETITION

BASE_URL = "https://huggingface.co/datasets/eatpizzanot/soccer-dataset/resolve/main"
FILES = {
    "fixtures": f"{BASE_URL}/fixtures.parquet",
    "leagues": f"{BASE_URL}/leagues.parquet",
    "teams": f"{BASE_URL}/teams.parquet",
    "match_stats": f"{BASE_URL}/match_stats.parquet",
    "fixture_players": f"{BASE_URL}/fixture_players.parquet",
    "fixture_players_stats_flat": f"{BASE_URL}/fixture_players_stats_flat.parquet",
}

PLAYER_AGGREGATES = (
    "rating_mean",
    "starter_rating_mean",
    "minutes_total",
    "goals_assists_total",
    "shots_total",
    "shots_on_total",
    "passes_key_total",
    "duels_won_total",
    "tackles_total",
    "cards_yellow_total",
    "cards_red_total",
    "penalty_scored_total",
    "penalty_missed_total",
    "lineup_size",
    "starter_count",
    "player_stat_coverage",
)


def _download(url: str, path: Path, *, timeout: int = 120, retries: int = 3) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    last_error: Exception | None = None
    for attempt in range(1, max(1, int(retries)) + 1):
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            response = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "SoccerPredictionResearch/1.0", "Accept": "application/octet-stream"},
                stream=True,
            )
            response.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
            tmp.replace(path)
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception as exc:
            last_error = exc
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
            if attempt < retries:
                time.sleep(float(attempt) * 2.0)
    raise RuntimeError(f"failed to download {url}") from last_error


def _read_parquet_batches(path: Path, columns: list[str], batch_size: int = 100_000):
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for the streaming player-history adapter") from exc
    dataset = ds.dataset(path, format="parquet")
    scanner = dataset.scanner(columns=columns, batch_size=batch_size)
    for batch in scanner.to_batches():
        yield batch.to_pandas()


def _aggregate_lineups(path: Path, fixture_ids: set[int]) -> pd.DataFrame:
    rows = []
    columns = ["id", "fixture_id", "team_id", "player_id", "is_starter", "minutes", "rating"]
    for batch in _read_parquet_batches(path, columns):
        batch["fixture_id"] = pd.to_numeric(batch["fixture_id"], errors="coerce")
        batch["team_id"] = pd.to_numeric(batch["team_id"], errors="coerce")
        batch = batch[batch["fixture_id"].isin(fixture_ids)].copy()
        if batch.empty:
            continue
        batch["is_starter"] = batch["is_starter"].astype(bool)
        batch["minutes"] = pd.to_numeric(batch["minutes"], errors="coerce")
        batch["rating"] = pd.to_numeric(batch["rating"], errors="coerce")
        batch["starter_rating"] = batch["rating"].where(batch["is_starter"])
        grouped = batch.groupby(["fixture_id", "team_id"], as_index=False).agg(
            lineup_size=("player_id", "count"),
            starter_count=("is_starter", "sum"),
            rating_sum=("rating", "sum"),
            rating_n=("rating", "count"),
            starter_rating_sum=("starter_rating", "sum"),
            starter_rating_n=("starter_rating", "count"),
            minutes_total=("minutes", "sum"),
        )
        rows.append(grouped)
    if not rows:
        return pd.DataFrame(columns=[
            "fixture_id", "team_id", "lineup_size", "starter_count", "rating_sum",
            "rating_n", "starter_rating_sum", "starter_rating_n", "minutes_total"
        ])
    out = pd.concat(rows, ignore_index=True)
    out = out.groupby(["fixture_id", "team_id"], as_index=False).agg({
        "lineup_size": "sum",
        "starter_count": "sum",
        "rating_sum": "sum",
        "rating_n": "sum",
        "starter_rating_sum": "sum",
        "starter_rating_n": "sum",
        "minutes_total": "sum",
    })
    out["rating_mean"] = out["rating_sum"] / out["rating_n"].replace(0, np.nan)
    out["starter_rating_mean"] = out["starter_rating_sum"] / out["starter_rating_n"].replace(0, np.nan)
    return out.drop(columns=["rating_sum", "rating_n", "starter_rating_sum", "starter_rating_n"])


def _aggregate_player_stats(path: Path, fixture_ids: set[int]) -> pd.DataFrame:
    rows = []
    columns = [
        "fixture_player_id", "fixture_id", "player_id",
        "games_rating", "games_minutes", "goals_assists", "shots_total", "shots_on",
        "passes_key", "duels_won", "tackles_total", "cards_yellow", "cards_red",
        "penalty_scored", "penalty_missed",
    ]
    for batch in _read_parquet_batches(path, columns):
        batch["fixture_id"] = pd.to_numeric(batch["fixture_id"], errors="coerce")
        batch = batch[batch["fixture_id"].isin(fixture_ids)].copy()
        if batch.empty:
            continue
        numeric = [
            "games_rating", "games_minutes", "goals_assists", "shots_total", "shots_on",
            "passes_key", "duels_won", "tackles_total", "cards_yellow", "cards_red",
            "penalty_scored", "penalty_missed",
        ]
        for c in numeric:
            batch[c] = pd.to_numeric(batch[c], errors="coerce")
        grouped = batch.groupby(["fixture_id", "player_id"], as_index=False).agg(
            rating=("games_rating", "mean"),
            minutes=("games_minutes", "sum"),
            goals_assists=("goals_assists", "sum"),
            shots_total=("shots_total", "sum"),
            shots_on_total=("shots_on", "sum"),
            passes_key_total=("passes_key", "sum"),
            duels_won_total=("duels_won", "sum"),
            tackles_total=("tackles_total", "sum"),
            cards_yellow_total=("cards_yellow", "sum"),
            cards_red_total=("cards_red", "sum"),
            penalty_scored_total=("penalty_scored", "sum"),
            penalty_missed_total=("penalty_missed", "sum"),
        )
        rows.append(grouped)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_global_player_history(
    *,
    start_year: int = 2012,
    end_year: int = 2025,
    cache_dir: str = "cache/global_player_history",
) -> tuple[pd.DataFrame, dict]:
    cache = Path(cache_dir)
    hashes = {
        name: _download(url, cache / f"{name}.parquet")
        for name, url in FILES.items()
    }

    fixtures = pd.read_parquet(
        cache / "fixtures.parquet",
        columns=["id", "date_utc", "league_id", "home_team_id", "away_team_id"],
        engine="pyarrow",
    )
    leagues = pd.read_parquet(cache / "leagues.parquet", columns=["id", "name"], engine="pyarrow")
    teams = pd.read_parquet(cache / "teams.parquet", columns=["id", "name"], engine="pyarrow")
    match_stats = pd.read_parquet(
        cache / "match_stats.parquet",
        columns=["fixture_id", "known_at"],
        engine="pyarrow",
    )

    fixtures["date_utc"] = pd.to_datetime(fixtures["date_utc"], utc=True, errors="coerce")
    fixtures["league_id"] = pd.to_numeric(fixtures["league_id"], errors="coerce")
    fixtures = fixtures[
        fixtures["date_utc"].notna()
        & fixtures["date_utc"].dt.year.between(int(start_year), int(end_year))
    ].copy()

    leagues["competition"] = leagues["name"].map(LEAGUE_TO_COMPETITION)
    leagues = leagues[
        leagues["competition"].notna()
        & leagues["competition"].astype(str).isin(TARGET_COMPETITIONS)
    ]
    meta = fixtures.merge(
        leagues[["id", "competition"]].rename(columns={"id": "league_id"}),
        on="league_id",
        how="inner",
        validate="many_to_one",
    )
    match_stats["known_at"] = pd.to_datetime(match_stats["known_at"], utc=True, errors="coerce")
    match_stats["fixture_id"] = pd.to_numeric(match_stats["fixture_id"], errors="coerce")
    if match_stats["fixture_id"].duplicated().any():
        raise RuntimeError("global player history match_stats contains duplicate fixture_id values")
    meta = meta.merge(match_stats, left_on="id", right_on="fixture_id", how="inner", validate="one_to_one")
    meta = meta[meta["known_at"].notna()].copy()

    teams["id"] = pd.to_numeric(teams["id"], errors="coerce")
    meta["home_team_id"] = pd.to_numeric(meta["home_team_id"], errors="coerce")
    meta["away_team_id"] = pd.to_numeric(meta["away_team_id"], errors="coerce")
    meta = meta.merge(
        teams.rename(columns={"id": "home_team_id", "name": "home_team"}),
        on="home_team_id", how="left", validate="many_to_one",
    ).merge(
        teams.rename(columns={"id": "away_team_id", "name": "away_team"}),
        on="away_team_id", how="left", validate="many_to_one",
    )
    if meta[["home_team", "away_team"]].isna().any().any():
        raise RuntimeError("global player history team entity resolution is incomplete")

    fixture_ids = set(pd.to_numeric(meta["id"], errors="coerce").dropna().astype(int).tolist())
    lineup = _aggregate_lineups(cache / "fixture_players.parquet", fixture_ids)
    stats = _aggregate_player_stats(cache / "fixture_players_stats_flat.parquet", fixture_ids)

    if not stats.empty:
        stats["fixture_id"] = pd.to_numeric(stats["fixture_id"], errors="coerce")
        team_rows = []
        for batch in _read_parquet_batches(
            cache / "fixture_players.parquet",
            ["fixture_id", "team_id", "player_id"],
        ):
            batch["fixture_id"] = pd.to_numeric(batch["fixture_id"], errors="coerce")
            batch = batch[batch["fixture_id"].isin(fixture_ids)].copy()
            if batch.empty:
                continue
            team_rows.append(batch[["fixture_id", "team_id", "player_id"]])
        if team_rows:
            identity = pd.concat(team_rows, ignore_index=True).drop_duplicates(
                ["fixture_id", "player_id"], keep="last"
            )
            stats = stats.merge(identity, on=["fixture_id", "player_id"], how="left", validate="many_to_one")
        else:
            stats["team_id"] = np.nan
        stats = stats.dropna(subset=["team_id"]).copy()
        team_stats = stats.groupby(["fixture_id", "team_id"], as_index=False).agg({
            "rating": "mean",
            "minutes": "sum",
            "goals_assists": "sum",
            "shots_total": "sum",
            "shots_on_total": "sum",
            "passes_key_total": "sum",
            "duels_won_total": "sum",
            "tackles_total": "sum",
            "cards_yellow_total": "sum",
            "cards_red_total": "sum",
            "penalty_scored_total": "sum",
            "penalty_missed_total": "sum",
        }).rename(columns={
            "rating": "player_stat_rating_mean",
            "minutes": "player_stat_minutes_total",
        })
        coverage = stats.groupby(["fixture_id", "team_id"], as_index=False).size().rename(columns={"size": "player_stat_coverage"})
        team_stats = team_stats.merge(coverage, on=["fixture_id", "team_id"], how="left", validate="one_to_one")
    else:
        team_stats = pd.DataFrame()

    team = lineup.merge(
        team_stats,
        on=["fixture_id", "team_id"],
        how="outer",
        validate="one_to_one",
        suffixes=("", "_stats"),
    )

    meta_ids = meta[["id", "home_team_id", "away_team_id"]].rename(columns={"id": "fixture_id"})
    team = team.merge(meta_ids, on="fixture_id", how="inner", validate="many_to_one")
    team["side"] = np.select(
        [team["team_id"] == team["home_team_id"], team["team_id"] == team["away_team_id"]],
        ["home", "away"],
        default="other",
    )
    team = team[team["side"].isin(["home", "away"])].copy()

    feature_map = {
        "rating_mean": "rating_mean",
        "starter_rating_mean": "starter_rating_mean",
        "minutes_total": "minutes_total",
        "goals_assists_total": "goals_assists",
        "shots_total": "shots_total",
        "shots_on_total": "shots_on_total",
        "passes_key_total": "passes_key_total",
        "duels_won_total": "duels_won_total",
        "tackles_total": "tackles_total",
        "cards_yellow_total": "cards_yellow_total",
        "cards_red_total": "cards_red_total",
        "penalty_scored_total": "penalty_scored_total",
        "penalty_missed_total": "penalty_missed_total",
        "lineup_size": "lineup_size",
        "starter_count": "starter_count",
        "player_stat_coverage": "player_stat_coverage",
    }

    wide = meta[["id", "competition", "date_utc", "home_team", "away_team", "known_at"]].copy()
    wide = wide.rename(columns={"id": "fixture_id"})
    for key, source in feature_map.items():
        if source not in team.columns:
            continue
        pivot = team.pivot(index="fixture_id", columns="side", values=source)
        if "home" in pivot:
            wide = wide.merge(
                pivot["home"].rename(f"home_player_{key}"),
                on="fixture_id", how="left", validate="one_to_one"
            )
        if "away" in pivot:
            wide = wide.merge(
                pivot["away"].rename(f"away_player_{key}"),
                on="fixture_id", how="left", validate="one_to_one"
            )

    wide["source_available_at_utc"] = wide["known_at"]
    wide["retrieved_at_utc"] = pd.Timestamp.now(tz="UTC")
    wide["source_name"] = "Global Football (Soccer) Data Lake / player tables"
    wide["pit_verified"] = True
    wide["match_id"] = "gfdl-player:" + wide["fixture_id"].astype("string")
    wide["kickoff_utc"] = wide["date_utc"]

    result_columns = [
        "match_id", "competition", "kickoff_utc", "home_team", "away_team",
        "source_available_at_utc", "retrieved_at_utc", "source_name", "pit_verified",
        *[c for c in wide.columns if c.startswith("home_player_") or c.startswith("away_player_")],
    ]
    result = wide[result_columns].sort_values(
        ["competition", "kickoff_utc", "match_id"], kind="mergesort"
    ).reset_index(drop=True)

    coverage = result.groupby("competition", dropna=False).size().rename("rows").reset_index()
    status = {
        "status": "OK",
        "rows": int(len(result)),
        "coverage": coverage.to_dict(orient="records"),
        "pit_basis": "match_stats.known_at",
        "production_status": "RESEARCH_ONLY",
        "source_license": "CC-BY-4.0",
        "source_url": "https://huggingface.co/datasets/eatpizzanot/soccer-dataset",
        "snapshot_hashes": hashes,
    }
    return result, status


def save_research_artifacts(
    *,
    start_year: int,
    end_year: int,
    output_dir: str = "artifacts/global_player_history",
    cache_dir: str = "cache/global_player_history",
) -> dict:
    import json
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame, status = load_global_player_history(
        start_year=start_year, end_year=end_year, cache_dir=cache_dir
    )
    frame.to_csv(out / "player_history_features.csv", index=False)
    status = {**status, "feature_artifact": str(out / "player_history_features.csv")}
    (out / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return status
