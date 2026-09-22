from __future__ import annotations

"""Fail-closed PIT evidence from versioned Football-Data cache snapshots for Serie A."""

import hashlib
import json
import os
import time
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.data.pit_source_adapter_v2 import _result_lower_bound

GITHUB_API = "https://api.github.com"
REPOSITORY = "footballcsv/cache.footballdata"
COMPETITION = "SA"

SNAPSHOT_PATHS = {
    2019: "2019-20/it.1.csv",
    2020: "2020-21/it.1.csv",
}


def _utc(value: Any):
    if value is None or value == "":
        return None
    try:
        stamp = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC").to_pydatetime()


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "SoccerPredictionResearch/footballcsv-SA-PIT",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _request_json(url: str, timeout: int = 30) -> Any:
    for attempt in range(1, 4):
        try:
            response = requests.get(url, headers=_headers(), timeout=timeout)
            if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
                raise RuntimeError("github_api_rate_limit_exhausted")
            if response.status_code in {429, 500, 502, 503, 504} and attempt < 3:
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else float(attempt * 2)
                except ValueError:
                    delay = float(attempt * 2)
                time.sleep(min(30.0, max(1.0, delay)))
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            if attempt >= 3:
                raise RuntimeError(f"github_request_failed:{type(exc).__name__}:{exc}") from exc
            time.sleep(float(attempt * 2))
    raise RuntimeError("github_request_failed_after_retries")


def _cache_file(cache_dir: str, prefix: str, *parts: str) -> Path:
    key = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]
    suffix = ".json" if prefix == "commits" else ".csv"
    path = Path(cache_dir) / f"footballcsv-italy-{prefix}-{key}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _commit_time(commit: dict[str, Any]):
    return _utc(((commit.get("commit") or {}).get("committer") or {}).get("date"))


def _commits(path: str, cache_dir: str, timeout: int) -> list[dict[str, Any]]:
    cache = _cache_file(cache_dir, "commits", REPOSITORY, path)
    refresh_hours = float(os.getenv("PIT_FOOTBALLCSV_ITALY_COMMIT_CACHE_REFRESH_HOURS", "6"))
    if cache.exists() and refresh_hours > 0:
        try:
            if (time.time() - cache.stat().st_mtime) <= refresh_hours * 3600.0:
                payload = json.loads(cache.read_text(encoding="utf-8"))
                if isinstance(payload, list) and payload:
                    return payload
        except Exception:
            pass
    payload = _request_json(
        f"{GITHUB_API}/repos/{REPOSITORY}/commits?path={path}&per_page=100&page=1",
        timeout,
    )
    if not isinstance(payload, list):
        raise RuntimeError("invalid_footballcsv_italy_commit_history")
    if payload:
        cache.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def _snapshot(path: str, sha: str, cache_dir: str, timeout: int) -> str:
    cache = _cache_file(cache_dir, "snapshot", REPOSITORY, path, sha)
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    response = requests.get(
        f"https://raw.githubusercontent.com/{REPOSITORY}/{sha}/{path}",
        headers={"User-Agent": "SoccerPredictionResearch/footballcsv-SA-PIT"},
        timeout=timeout,
    )
    response.raise_for_status()
    text = response.content.decode("utf-8", errors="replace")
    if text.lstrip().lower().startswith(("<!doctype html", "<html")):
        raise ValueError("html_instead_of_snapshot")
    cache.write_text(text, encoding="utf-8")
    return text


def _norm(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    return "".join(ch.casefold() for ch in text if ch.isalnum())


def _target_key(row: pd.Series) -> tuple | None:
    try:
        return (
            pd.Timestamp(row["kickoff_utc"]).date().isoformat(),
            _norm(row["home_team"]),
            _norm(row["away_team"]),
            int(float(row["home_goals"])),
            int(float(row["away_goals"])),
        )
    except (TypeError, ValueError, KeyError):
        return None


def _source_rows(text: str) -> dict[tuple, int]:
    frame = pd.read_csv(pd.io.common.StringIO(text), low_memory=False)
    required = {"Date", "Team 1", "FT", "Team 2"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"footballcsv Italy snapshot missing columns: {sorted(required - set(frame.columns))}"
        )
    counts: dict[tuple, int] = {}
    for row in frame.itertuples(index=False):
        try:
            date = pd.Timestamp(getattr(row, "Date")).date().isoformat()
            home = _norm(getattr(row, "Team_1"))
            away = _norm(getattr(row, "Team_2"))
            ft = str(getattr(row, "FT") or "").strip()
            if "-" not in ft:
                continue
            left, right = ft.split("-", 1)
            hg = int(float(left))
            ag = int(float(right))
        except (TypeError, ValueError):
            continue
        key = (date, home, away, hg, ag)
        counts[key] = counts.get(key, 0) + 1
    return counts


def apply_footballcsv_italy_pit(
    history: pd.DataFrame,
    *,
    cache_dir: str = "data/raw/pit_evidence",
    timeout: int = 30,
) -> pd.DataFrame:
    if history is None or history.empty:
        return history.copy() if history is not None else history

    mask = history["competition"].astype(str).eq(COMPETITION)
    if not mask.any():
        return history.iloc[0:0].copy()

    verified_rows: list[dict[str, Any]] = []
    for season_start, group in history.loc[mask].groupby("season_start", sort=True):
        try:
            year = int(season_start)
        except (TypeError, ValueError):
            continue
        path = SNAPSHOT_PATHS.get(year)
        if not path:
            continue

        pending: dict[Any, tuple[tuple, Any]] = {}
        for idx, row in group.iterrows():
            if str(row.get("pit_evidence_status", "")) == "VERIFIED":
                continue
            key = _target_key(row)
            lower_bound, _reason = _result_lower_bound(row)
            if key is not None and lower_bound is not None:
                pending[idx] = (key, lower_bound)
        if not pending:
            continue

        try:
            commits = _commits(path, cache_dir, timeout)
        except Exception:
            continue

        ordered = []
        for commit in commits:
            dt = _commit_time(commit)
            sha = str(commit.get("sha", "")).strip()
            if dt is not None and sha:
                ordered.append((dt, sha))
        ordered.sort()

        parsed_cache: dict[str, dict[tuple, int] | None] = {}
        for dt, sha in ordered:
            if not pending:
                break
            eligible = [
                idx for idx, (_, bound) in pending.items()
                if dt >= _utc(bound)
            ]
            if not eligible:
                continue
            if sha not in parsed_cache:
                try:
                    parsed_cache[sha] = _source_rows(_snapshot(path, sha, cache_dir, timeout))
                except Exception:
                    parsed_cache[sha] = None
            counts = parsed_cache[sha]
            if counts is None:
                continue
            for idx in list(eligible):
                key, _bound = pending[idx]
                if counts.get(key, 0) != 1:
                    continue
                verified_rows.append(
                    {
                        "source_available_at_utc": dt.isoformat(),
                        "pit_evidence_status": "VERIFIED",
                        "pit_evidence_reason": "immutable_footballcsv_italy_snapshot_contains_unique_exact_result_after_result_lower_bound",
                        "pit_evidence_url": f"https://github.com/{REPOSITORY}/blob/{sha}/{path}",
                        "capture_digest": sha,
                        "__index__": idx,
                    }
                )
                del pending[idx]

    if not verified_rows:
        return history.iloc[0:0].copy()
    out = pd.DataFrame(verified_rows).set_index("__index__")
    out.index.name = history.index.name
    return out
