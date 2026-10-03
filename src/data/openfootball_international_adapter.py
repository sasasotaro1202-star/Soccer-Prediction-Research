from __future__ import annotations

"""Expanded free historical international-football acquisition.

The public-domain openfootball/internationals repository contains tournament/year
files for major international competitions. We discover matching files from the
repository tree once, then download/cache only the requested seasons.
"""

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.data.http_resilience import resilient_get

from src.data.openfootball_adapter import parse_football_txt

TREE_URL = "https://api.github.com/repos/openfootball/internationals/git/trees/master?recursive=1"
RAW_BASE = "https://raw.githubusercontent.com/openfootball/internationals/master/"

# Active project competition codes -> openfootball tournament directory/name.
PATH_PATTERNS: dict[str, re.Pattern[str]] = {
    "UEFA_EURO_M": re.compile(r"^uefa_euro/(\d{4})_uefa_euro\.txt$"),
    "UEFA_EURO_QUALI_M": re.compile(r"^uefa_euro_qualification/(\d{4})_uefa_euro_qualification\.txt$"),
    "UEFA_NATIONS_LEAGUE_M": re.compile(r"^uefa_nations_league/(\d{4})_uefa_nations_league\.txt$"),
    "ASIAN_CUP": re.compile(r"^afc_asian_cup/(\d{4})_afc_asian_cup\.txt$"),
    "ASIAN_CUP_QUALI": re.compile(r"^afc_asian_cup_qualification/(\d{4})_afc_asian_cup_qualification\.txt$"),
    "WORLD_CUP": re.compile(r"^fifa_world_cup/(\d{4})_fifa_world_cup\.txt$"),
    "WORLD_CUP_QUALI": re.compile(r"^fifa_world_cup_qualification/(\d{4})_fifa_world_cup_qualification\.txt$"),
    "INTL_M": re.compile(r"^friendly/(\d{4})_friendly\.txt$"),
}

HEADERS = {"User-Agent": "SoccerPredictionResearch/1.0", "Accept": "application/vnd.github+json"}


def _tree_paths() -> list[str]:
    response = resilient_get(requests.get, TREE_URL, timeout=None, retries=6, backoff=2.0, headers=HEADERS)
    payload = response.json()
    if payload.get("truncated"):
        raise RuntimeError("openfootball international source tree is truncated; refusing incomplete discovery")
    return [
        str(item["path"])
        for item in payload.get("tree", [])
        if item.get("type") == "blob" and str(item.get("path", "")).endswith(".txt")
    ]


def _cached_tree_paths(cache_dir: str = "data/raw/openfootball-internationals") -> list[str]:
    """Recover a previously observed source tree without inventing new paths.

    Cached paths are only used as a transport fallback when live GitHub tree
    discovery is temporarily unavailable. The cached files themselves remain
    subject to the normal parser/PIT checks.
    """
    root = Path(cache_dir)
    if not root.is_dir():
        return []
    prefix = root.resolve()
    paths: list[str] = []
    for file_path in root.rglob("*.txt"):
        if not file_path.is_file() or file_path.stat().st_size <= 0:
            continue
        try:
            relative = file_path.resolve().relative_to(prefix).as_posix()
        except ValueError:
            continue
        if relative:
            paths.append(relative)
    return sorted(set(paths))


def _cache_path(cache_dir: str, relative_path: str) -> Path:
    path = Path(cache_dir) / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _fetch_text(relative_path: str, cache_dir: str) -> tuple[str, bytes, str]:
    url = RAW_BASE + relative_path
    cache = _cache_path(cache_dir, relative_path)
    raw = cache.read_bytes() if cache.exists() and cache.stat().st_size > 0 else None
    if raw is None:
        response = resilient_get(requests.get, url, timeout=None, retries=6, backoff=2.0, headers={"User-Agent": HEADERS["User-Agent"]})
        raw = response.content
        cache.write_bytes(raw)
    return raw.decode("utf-8", errors="replace"), raw, url


def _season_ok(season_start: int, start_year: int, end_year: int) -> bool:
    return int(start_year) <= int(season_start) <= int(end_year)


def _one(item: tuple[str, str, int, int, str, str]) -> tuple[pd.DataFrame, dict]:
    competition, relative_path, season_start, start_year, end_year, cache_dir = item
    if not _season_ok(season_start, start_year, end_year):
        return pd.DataFrame(), {}
    try:
        text, raw, url = _fetch_text(relative_path, cache_dir)
        frame = parse_football_txt(
            text,
            competition,
            season_start,
            url,
            raw,
            calendar_year=competition in {"WORLD_CUP", "ASIAN_CUP", "INTL_M"},
        )
        # World Cup, Asian Cup and international friendlies use calendar-year labels.
        if competition in {"WORLD_CUP", "ASIAN_CUP", "INTL_M"} and not frame.empty:
            frame["season"] = str(season_start)
        coverage = {
            "competition": competition,
            "season": str(season_start),
            "status": "AVAILABLE" if not frame.empty else "UNAVAILABLE",
            "rows": int(len(frame)),
            "source": "openfootball/internationals",
            "source_path": relative_path,
        }
        return frame, coverage
    except Exception as exc:
        return pd.DataFrame(), {
            "competition": competition,
            "season": str(season_start),
            "status": "UNAVAILABLE",
            "rows": 0,
            "source": "openfootball/internationals",
            "source_path": relative_path,
            "error": str(exc),
        }


def load_openfootball_international_history(
    start_year: int = 2000,
    end_year: int = 2026,
    max_workers: int = 8,
    cache_dir: str = "data/raw/openfootball-internationals",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # GitHub API rate limits/temporary 403 responses must not suppress unrelated
    # historical sources. Prefer a previously observed path index when the live
    # tree cannot be read; without a cache, degrade to empty output and let the
    # downstream coverage/PIT gates record the resulting scope deficit.
    try:
        paths = _tree_paths()
        discovery_mode = "LIVE"
    except (requests.Timeout, requests.ConnectionError):
        paths = _cached_tree_paths(cache_dir)
        discovery_mode = "CACHE_FALLBACK" if paths else "UNAVAILABLE"
    except requests.HTTPError as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if int(status_code or 0) not in {403, 429, 500, 502, 503, 504}:
            raise
        paths = _cached_tree_paths(cache_dir)
        discovery_mode = "CACHE_FALLBACK" if paths else "UNAVAILABLE"
    tasks: list[tuple[str, str, int, int, str, str]] = []
    for competition, pattern in PATH_PATTERNS.items():
        for relative_path in paths:
            match = pattern.match(relative_path)
            if not match:
                continue
            season_start = int(match.group(1))
            if _season_ok(season_start, start_year, end_year):
                tasks.append((competition, relative_path, season_start, start_year, end_year, cache_dir))

    tasks.sort(key=lambda x: (x[0], x[2], x[1]))
    frames: list[pd.DataFrame] = []
    coverage: list[dict] = []
    if not tasks:
        return pd.DataFrame(), pd.DataFrame()

    with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), len(tasks)))) as pool:
        futures = [pool.submit(_one, task) for task in tasks]
        for future in as_completed(futures):
            frame, row = future.result()
            if not frame.empty:
                frames.append(frame)
            if row:
                row["discovery_mode"] = discovery_mode
                coverage.append(row)

    history = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not history.empty:
        history = history.drop_duplicates(
            ["competition", "season_start", "kickoff_utc", "home_team", "away_team"],
            keep="first",
        ).sort_values(
            ["competition", "kickoff_utc", "home_team", "away_team"],
            kind="mergesort",
        ).reset_index(drop=True)

    coverage_frame = pd.DataFrame(coverage).sort_values(
        ["competition", "season", "source_path"],
        kind="mergesort",
    ).reset_index(drop=True) if coverage else pd.DataFrame()
    return history, coverage_frame
