"""Research-only adapter for the open Global Football (Soccer) Data Lake.

The dataset publishes a `known_at` timestamp for post-match facts. We use that
timestamp as a conservative feature-availability boundary and never treat
retrieval time as historical availability.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pandas as pd
import requests

from src.data.competition_catalog import ACTIVE_SCOPE

BASE_URL = "https://huggingface.co/datasets/eatpizzanot/soccer-dataset/resolve/main"
FILES = {
    "fixtures": f"{BASE_URL}/fixtures.parquet",
    "teams": f"{BASE_URL}/teams.parquet",
    "leagues": f"{BASE_URL}/leagues.parquet",
    "match_stats": f"{BASE_URL}/match_stats.parquet",
}

# Start conservatively with competitions already represented by the current
# production-research feature stack. Additional mappings can be added only after
# entity, PIT and OOS checks.
LEAGUE_TO_COMPETITION = {
    "Premier League": "EPL",
    "Bundesliga": "BL1",
    "Serie A": "SA",
    "La Liga": "LL",
    "Ligue 1": "FL1",
    "Eredivisie": "ERE",
    "J1 League": "J1",
    "J2 League": "J2",
    "J3 League": "J3",
    "UEFA Champions League": "UCL",
    "UEFA Europa League": "UEL",
    "UEFA Conference League": "UECL",
    "UEFA Women's Champions League": "UWCL",
}

FIXTURE_COLUMNS = (
    "id",
    "date_utc",
    "league_id",
    "home_team_id",
    "away_team_id",
    "goals_home",
    "goals_away",
    "known_at",
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
                headers={
                    "User-Agent": "Soccer-Prediction-Research/1.0",
                    "Accept": "application/octet-stream",
                },
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


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")


def load_global_datalake_history(
    *,
    start_year: int = 2012,
    end_year: int = 2025,
    cache_dir: str = "cache/global_datalake",
) -> tuple[pd.DataFrame, dict]:
    """Load mapped historical rows with conservative PIT timestamps.

    This adapter is intentionally separate from the production history loader.
    It returns research evidence; callers must still run the project's normal
    feature/OOS/adoption gates before any production use.
    """
    cache = Path(cache_dir)
    fixture_hash = _download(FILES["fixtures"], cache / "fixtures.parquet")
    team_hash = _download(FILES["teams"], cache / "teams.parquet")
    league_hash = _download(FILES["leagues"], cache / "leagues.parquet")
    stats_hash = _download(FILES["match_stats"], cache / "match_stats.parquet")

    fixtures = pd.read_parquet(cache / "fixtures.parquet", engine="pyarrow")
    teams = pd.read_parquet(cache / "teams.parquet", engine="pyarrow")
    leagues = pd.read_parquet(cache / "leagues.parquet", engine="pyarrow")
    match_stats = pd.read_parquet(cache / "match_stats.parquet", engine="pyarrow")

    _require_columns(
        fixtures,
        {"id", "date_utc", "league_id", "home_team_id", "away_team_id", "goals_home", "goals_away", "known_at"},
        "global_datalake fixtures",
    )
    _require_columns(leagues, {"id", "name"}, "global_datalake leagues")
    _require_columns(teams, {"id", "name"}, "global_datalake teams")

    selected = fixtures[[c for c in FIXTURE_COLUMNS if c in fixtures.columns]].copy()
    selected["date_utc"] = pd.to_datetime(selected["date_utc"], utc=True, errors="coerce")
    selected["known_at"] = pd.to_datetime(selected["known_at"], utc=True, errors="coerce")
    selected = selected[
        selected["date_utc"].notna()
        & selected["known_at"].notna()
        & selected["goals_home"].notna()
        & selected["goals_away"].notna()
    ].copy()
    selected = selected[
        selected["date_utc"].dt.year.between(int(start_year), int(end_year))
    ].copy()

    league_map = leagues[["id", "name"]].copy()
    league_map["competition"] = league_map["name"].map(LEAGUE_TO_COMPETITION)
    league_map = league_map[league_map["competition"].notna()]
    league_map = league_map[league_map["competition"].astype(str).isin(ACTIVE_SCOPE)]
    selected = selected.merge(
        league_map[["id", "name", "competition"]].rename(columns={"id": "league_id"}),
        on="league_id",
        how="inner",
        validate="many_to_one",
    )

    # Match statistics are post-match facts governed by the same fixture-level
    # known_at boundary. Join only the documented side-level aggregates needed by
    # the PIT feature layer; closing odds are deliberately not consumed here.
    stats_columns = [
        "fixture_id",
        "home_shots_total", "away_shots_total",
        "home_shots_on_goal", "away_shots_on_goal",
        "home_shots_inside_box", "away_shots_inside_box",
        "home_shots_outside_box", "away_shots_outside_box",
        "home_blocked_shots", "away_blocked_shots",
        "home_penalties", "away_penalties",
        "home_corners", "away_corners",
        "home_yellow_cards", "away_yellow_cards",
        "home_red_cards", "away_red_cards",
        "home_xg", "away_xg",
        "home_possession", "away_possession",
        "home_fouls", "away_fouls",
        "home_offsides", "away_offsides",
        "home_pass_accuracy", "away_pass_accuracy",
        "home_goals_ht", "away_goals_ht",
        "home_xg_ht", "away_xg_ht",
    ]
    stats = match_stats[[c for c in stats_columns if c in match_stats.columns]].copy()
    if stats["fixture_id"].duplicated().any():
        raise RuntimeError("global_datalake match_stats contains duplicate fixture_id values")
    selected = selected.merge(stats, left_on="id", right_on="fixture_id", how="left", validate="one_to_one")
    if "fixture_id" in selected.columns:
        selected = selected.drop(columns=["fixture_id"])

    # Normalize provider-specific names to the canonical feature schema used by
    # src.features.soccer_features. Prefer Global Data Lake match_stats values,
    # while leaving missing observations missing rather than fabricating zeros.
    STANDARD_STAT_ALIASES = {
        "home_shots": "home_shots_total",
        "away_shots": "away_shots_total",
        "home_shots_on_target": "home_shots_on_goal",
        "away_shots_on_target": "away_shots_on_goal",
        "home_corners": "home_corners",
        "away_corners": "away_corners",
        "home_fouls": "home_fouls",
        "away_fouls": "away_fouls",
        "home_yellow_cards": "home_yellow_cards",
        "away_yellow_cards": "away_yellow_cards",
        "home_red_cards": "home_red_cards",
        "away_red_cards": "away_red_cards",
    }
    for canonical, provider_name in STANDARD_STAT_ALIASES.items():
        if provider_name in selected.columns:
            selected[canonical] = selected[provider_name]

    team_map = teams[["id", "name"]].copy()
    team_map["id"] = pd.to_numeric(team_map["id"], errors="coerce")
    selected["home_team_id"] = pd.to_numeric(selected["home_team_id"], errors="coerce")
    selected["away_team_id"] = pd.to_numeric(selected["away_team_id"], errors="coerce")
    selected = selected.merge(
        team_map.rename(columns={"id": "home_team_id", "name": "home_team"}),
        on="home_team_id",
        how="left",
        validate="many_to_one",
    )
    selected = selected.merge(
        team_map.rename(columns={"id": "away_team_id", "name": "away_team"}),
        on="away_team_id",
        how="left",
        validate="many_to_one",
    )
    selected["home_team"] = selected["home_team"].astype("string").str.strip()
    selected["away_team"] = selected["away_team"].astype("string").str.strip()
    if selected[["home_team", "away_team"]].isna().any().any():
        raise RuntimeError("global_datalake team entity resolution is incomplete")

    selected["match_id"] = (
        "gfdl:"
        + selected["id"].astype("string")
    )
    selected["kickoff_utc"] = selected["date_utc"]
    selected["home_goals"] = pd.to_numeric(selected["goals_home"], errors="coerce")
    selected["away_goals"] = pd.to_numeric(selected["goals_away"], errors="coerce")
    selected["source_available_at_utc"] = selected["known_at"]
    selected["retrieved_at_utc"] = pd.Timestamp.now(tz="UTC")
    selected["source_name"] = "Global Football (Soccer) Data Lake"
    selected["source_record_id"] = selected["id"].astype("string")
    selected["pit_verified"] = True
    selected["event_time_precision"] = "MINUTE"

    # Preserve the advanced side-level performance columns under the names
    # consumed by src.features.soccer_features.
    for column in (
        "home_xg", "away_xg",
        "home_possession", "away_possession",
        "home_fouls", "away_fouls",
        "home_offsides", "away_offsides",
        "home_pass_accuracy", "away_pass_accuracy",
        "home_goals_ht", "away_goals_ht",
        "home_xg_ht", "away_xg_ht",
        "home_shots_inside_box", "away_shots_inside_box",
        "home_shots_outside_box", "away_shots_outside_box",
        "home_blocked_shots", "away_blocked_shots",
        "home_penalties", "away_penalties",
    ):
        if column not in selected:
            selected[column] = pd.NA

    keep = [
        "match_id",
        "competition",
        "kickoff_utc",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "source_available_at_utc",
        "retrieved_at_utc",
        "source_name",
        "source_record_id",
        "pit_verified",
        "event_time_precision",
        "home_xg",
        "away_xg",
        "home_possession",
        "away_possession",
        "home_fouls",
        "away_fouls",
        "home_offsides",
        "away_offsides",
        "home_pass_accuracy",
        "away_pass_accuracy",
        "home_goals_ht",
        "away_goals_ht",
        "home_xg_ht",
        "away_xg_ht",
        "home_shots_inside_box",
        "away_shots_inside_box",
        "home_shots_outside_box",
        "away_shots_outside_box",
        "home_blocked_shots",
        "away_blocked_shots",
        "home_penalties",
        "away_penalties",
    ]
    result = selected[keep].copy()
    result = result.sort_values(
        ["competition", "kickoff_utc", "home_team", "away_team", "match_id"],
        kind="mergesort",
    ).drop_duplicates(
        subset=["competition", "kickoff_utc", "home_team", "away_team"],
        keep="first",
    ).reset_index(drop=True)

    coverage = (
        result.groupby("competition", dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values("competition", kind="mergesort")
    )
    status = {
        "status": "OK",
        "rows": int(len(result)),
        "coverage": coverage.to_dict(orient="records"),
        "pit_basis": "known_at",
        "source_license": "CC-BY-4.0",
        "source_url": "https://huggingface.co/datasets/eatpizzanot/soccer-dataset",
        "snapshot_hashes": {
            "fixtures.parquet": fixture_hash,
            "teams.parquet": team_hash,
            "leagues.parquet": league_hash,
            "match_stats.parquet": stats_hash,
        },
        "production_status": "RESEARCH_ONLY",
    }
    return result, status
