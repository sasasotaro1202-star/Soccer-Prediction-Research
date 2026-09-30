from __future__ import annotations

"""Bounded adaptive acquisition/discovery policy for soccer research.

The policy is deliberately fail-closed:
- only existing free/public adapters are auto-ingested;
- newly discovered sources remain candidates until their schema and PIT are verified;
- each run can widen the historical window and re-run adapters;
- repeated zero-yield rounds stop instead of creating an unbounded hot loop.
"""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


@dataclass(frozen=True)
class AcquisitionConfig:
    rounds: int = 4
    expand_back_years: int = 3
    expand_forward_years: int = 1
    minimum_start_year: int = 1990
    maximum_end_year: int = 2026
    max_discovery_competitions: int = 8
    discovery_per_competition: int = 4
    no_progress_stop_rounds: int = 1
    minimum_rows_per_competition: int = 300
    minimum_seasons_per_competition: int = 2


def load_config() -> AcquisitionConfig:
    now_year = datetime.now(timezone.utc).year

    def integer(name: str, default: int, minimum: int = 0) -> int:
        try:
            value = int(os.getenv(name, str(default)))
        except ValueError:
            value = default
        return max(value, minimum)

    return AcquisitionConfig(
        rounds=integer("SOCCER_ADAPTIVE_ACQUISITION_ROUNDS", 4, 1),
        expand_back_years=integer("SOCCER_ADAPTIVE_EXPAND_BACK_YEARS", 3, 0),
        expand_forward_years=integer("SOCCER_ADAPTIVE_EXPAND_FORWARD_YEARS", 1, 0),
        minimum_start_year=integer("SOCCER_ADAPTIVE_MIN_START_YEAR", 1990, 1950),
        maximum_end_year=integer("SOCCER_ADAPTIVE_MAX_END_YEAR", now_year, 1950),
        max_discovery_competitions=integer("SOCCER_ADAPTIVE_MAX_DISCOVERY_COMPETITIONS", 8, 1),
        discovery_per_competition=integer("SOCCER_ADAPTIVE_DISCOVERY_PER_COMPETITION", 4, 1),
        no_progress_stop_rounds=integer("SOCCER_ADAPTIVE_NO_PROGRESS_STOP_ROUNDS", 1, 0),
        minimum_rows_per_competition=integer("SOCCER_ADAPTIVE_MIN_ROWS_PER_COMPETITION", 300, 0),
        minimum_seasons_per_competition=integer("SOCCER_ADAPTIVE_MIN_SEASONS_PER_COMPETITION", 2, 0),
    )


def expanded_window(
    start_year: int,
    end_year: int,
    round_index: int,
    config: AcquisitionConfig,
) -> tuple[int, int]:
    back = int(config.expand_back_years) * int(round_index)
    forward = int(config.expand_forward_years) * int(round_index)
    return (
        max(int(config.minimum_start_year), int(start_year) - back),
        min(int(config.maximum_end_year), int(end_year) + forward),
    )


def deduplicate_history(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history.copy()
    out = history.copy()
    identity_cols = [
        c for c in (
            "competition",
            "season_start",
            "kickoff_utc",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "source_name",
        )
        if c in out.columns
    ]
    if not identity_cols:
        return out.reset_index(drop=True)
    return out.drop_duplicates(identity_cols, keep="first").reset_index(drop=True)


def select_preferred_sources(coverage: pd.DataFrame) -> pd.DataFrame:
    """Annotate a preferred observed source without discarding alternate evidence."""
    if coverage.empty or "competition" not in coverage.columns:
        return coverage.copy()

    work = coverage.copy()
    work["rows"] = pd.to_numeric(work.get("rows", 0), errors="coerce").fillna(0)
    work["status"] = work["status"].astype(str) if "status" in work.columns else "UNKNOWN"
    work["_pit_bonus"] = work["pit_capable"].astype(bool).astype(int) if "pit_capable" in work.columns else 0
    work["_rows_numeric"] = pd.to_numeric(work["rows"], errors="coerce").fillna(0.0)
    grouped = (
        work.groupby(["competition", "source"], dropna=False, sort=True)
        .agg(total_rows=("_rows_numeric", "sum"), pit_capable=("_pit_bonus", "max"))
        .reset_index()
    )
    grouped["_rank"] = grouped["total_rows"] + grouped["pit_capable"] * 0.001
    preferred = (
        grouped.sort_values(["competition", "_rank"], ascending=[True, False], kind="mergesort")
        .drop_duplicates("competition", keep="first")[["competition", "source", "total_rows"]]
        .rename(columns={"source": "preferred_source", "total_rows": "preferred_source_rows"})
    )
    work = work.merge(preferred, on="competition", how="left")
    work["preferred_source"] = work["preferred_source"].fillna("")
    work["preferred_source_rows"] = pd.to_numeric(
        work["preferred_source_rows"], errors="coerce"
    ).fillna(0).astype(int)
    return work.drop(columns=["_pit_bonus", "_rows_numeric"], errors="ignore")


def identify_data_deficits(
    history: pd.DataFrame,
    target_competitions: tuple[str, ...],
    *,
    minimum_rows_per_competition: int = 300,
    minimum_seasons_per_competition: int = 2,
) -> dict[str, dict[str, int]]:
    """Return the active competition frontier still needing acquisition."""
    row_counts = (
        history.groupby("competition").size().to_dict()
        if not history.empty and "competition" in history.columns
        else {}
    )
    season_counts: dict[str, int] = {}
    if not history.empty and "competition" in history.columns:
        season_col = "season_start" if "season_start" in history.columns else "season"
        season_counts = history.groupby("competition")[season_col].nunique(dropna=True).to_dict()

    deficits: dict[str, dict[str, int]] = {}
    for competition in target_competitions:
        comp = str(competition)
        rows = int(row_counts.get(comp, 0))
        seasons = int(season_counts.get(comp, 0))
        if rows < int(minimum_rows_per_competition) or seasons < int(minimum_seasons_per_competition):
            deficits[comp] = {
                "rows": rows,
                "minimum_rows": int(minimum_rows_per_competition),
                "seasons": seasons,
                "minimum_seasons": int(minimum_seasons_per_competition),
            }
    return deficits


def rank_discovery_targets(
    history: pd.DataFrame,
    target_competitions: tuple[str, ...],
    *,
    limit: int = 8,
) -> list[str]:
    counts = (
        history.groupby("competition").size().to_dict()
        if not history.empty and "competition" in history.columns
        else {}
    )
    return sorted(
        (str(c) for c in target_competitions),
        key=lambda c: (int(counts.get(c, 0)), c),
    )[: int(limit)]


def discover_free_github_sources(
    competitions: list[str],
    *,
    per_competition: int = 4,
    timeout: int = 15,
) -> dict[str, Any]:
    """Search public GitHub repository metadata for candidate datasets.

    Candidates are never auto-ingested. This function only produces a discovery
    frontier that later validation/adapters may accept.
    """
    enabled = os.getenv("SOCCER_ENABLE_PUBLIC_SOURCE_DISCOVERY", "1").strip().lower()
    if enabled not in {"1", "true", "yes"}:
        return {
            "status": "DISABLED",
            "reason": "SOCCER_ENABLE_PUBLIC_SOURCE_DISCOVERY is not enabled",
            "competitions": {},
        }

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "SoccerPredictionResearch-source-discovery",
    }
    result: dict[str, Any] = {
        "status": "COMPLETED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "GitHub public repository search",
        "candidates_are_research_only": True,
        "competitions": {},
    }

    for competition in competitions:
        queries = (
            f"soccer {competition} csv results",
            f"football {competition} dataset",
        )
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for query in queries:
            try:
                response = requests.get(
                    "https://api.github.com/search/repositories",
                    params={"q": query, "per_page": int(per_competition)},
                    headers=headers,
                    timeout=timeout,
                )
                response.raise_for_status()
                payload = response.json()
                for item in payload.get("items", []):
                    full_name = str(item.get("full_name", "")).strip()
                    if not full_name or full_name in seen:
                        continue
                    seen.add(full_name)
                    items.append(
                        {
                            "full_name": full_name,
                            "html_url": item.get("html_url"),
                            "description": item.get("description"),
                            "default_branch": item.get("default_branch"),
                            "updated_at": item.get("updated_at"),
                            "stars": int(item.get("stargazers_count", 0) or 0),
                            "forks": int(item.get("forks_count", 0) or 0),
                        }
                    )
            except Exception as exc:
                result.setdefault("errors", []).append(
                    {
                        "competition": competition,
                        "query": query,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        result["competitions"][competition] = items[: int(per_competition)]

    if result.get("errors") and not any(result["competitions"].values()):
        result["status"] = "DEGRADED"
    return result


def write_state(
    output_path: str | Path,
    *,
    config: AcquisitionConfig,
    rounds: list[dict[str, Any]],
    discovery: dict[str, Any] | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "rounds": rounds,
        "discovery": discovery or {},
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)
