from __future__ import annotations

"""PIT evidence from immutable OpenFootball country repositories.

This provider is audit-only. A match is VERIFIED only when an immutable Git
commit at/after the conservative result-publication lower bound contains the
exact completed-result identity. Commit time is retained as the observed
publication proxy; no timing is inferred earlier than the commit.
"""

import base64
import hashlib
import json
import os
import time
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.data.openfootball_adapter import parse_football_txt
from src.data.pit_source_adapter_v2 import _result_lower_bound

GITHUB_API = "https://api.github.com"

SOURCE_CONFIG = {
    "EPL": {"repo": "openfootball/england", "file": "{season}/1-premierleague.txt"},
    "BL1": {"repo": "openfootball/deutschland", "file": "{season}/1-bundesliga.txt"},
    "LL": {"repo": "openfootball/espana", "file": "{season}/1-liga.txt"},
    "SA": {"repo": "openfootball/italy", "file": "{season}/1-seriea.txt"},
}

def _utc(value: Any):
    if value is None or value == "":
        return None
    try:
        dt = pd.Timestamp(value).to_pydatetime()
    except Exception:
        return None
    if dt.tzinfo is None:
        from datetime import timezone
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(dt.tzinfo).astimezone(__import__("datetime").timezone.utc)

def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "SoccerPredictionResearch/OpenFootballPIT",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

def _request_json(url: str, timeout: int = 30) -> Any:
    last = None
    for attempt in range(1, 4):
        try:
            r = requests.get(url, headers=_headers(), timeout=timeout)
            last = r.status_code
            if r.status_code == 403 and r.headers.get("X-RateLimit-Remaining") == "0":
                raise RuntimeError("github_api_rate_limit_exhausted")
            if r.status_code in {429, 500, 502, 503, 504} and attempt < 3:
                retry_after = r.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and str(retry_after).replace(".", "", 1).isdigit() else float(attempt * 2)
                time.sleep(min(30.0, max(1.0, delay)))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            if attempt >= 3:
                raise RuntimeError(f"github_request_failed:{type(exc).__name__}:{exc}") from exc
            time.sleep(float(attempt * 2))
    raise RuntimeError(f"github_request_failed_after_retries:last_status={last}")

def _cache_file(cache_dir: str, prefix: str, *parts: str) -> Path:
    key = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]
    suffix = ".json" if prefix == "commits" else ".txt"
    path = Path(cache_dir) / f"country-{prefix}-{key}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def _commits(repo: str, path: str, cache_dir: str) -> list[dict[str, Any]]:
    cache = _cache_file(cache_dir, "commits", repo, path)
    try:
        refresh_hours = float(os.getenv("PIT_OPENFOOTBALL_COUNTRY_COMMIT_CACHE_REFRESH_HOURS", "6"))
    except ValueError:
        refresh_hours = 6.0
    fresh = cache.exists() and refresh_hours > 0 and (time.time() - cache.stat().st_mtime) <= refresh_hours * 3600.0
    if fresh:
        try:
            payload = json.loads(cache.read_text(encoding="utf-8"))
            if isinstance(payload, list) and payload:
                return payload
        except Exception:
            pass
    payload = _request_json(
        f"{GITHUB_API}/repos/{repo}/commits?path={path}&per_page=100"
    )
    if not isinstance(payload, list):
        return []
    cache.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload

def _snapshot(repo: str, path: str, sha: str, cache_dir: str, timeout: int) -> str:
    cache = _cache_file(cache_dir, "snapshot", repo, path, sha)
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    url = f"https://raw.githubusercontent.com/{repo}/{sha}/{path}"
    last = None
    for attempt in range(1, 4):
        try:
            r = requests.get(url, headers={"User-Agent": "SoccerPredictionResearch/OpenFootballPIT"}, timeout=timeout)
            r.raise_for_status()
            text = r.content.decode("utf-8", errors="replace")
            if text.lstrip().lower().startswith(("<!doctype html", "<html")):
                raise ValueError("html_instead_of_snapshot")
            cache.write_text(text, encoding="utf-8")
            return text
        except (requests.RequestException, OSError, ValueError) as exc:
            last = exc
            if attempt < 3:
                time.sleep(float(attempt * 2))
    raise RuntimeError(f"snapshot_fetch_failed:{type(last).__name__}:{last}")

def _norm(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    return "".join(ch.casefold() for ch in text if ch.isalnum())

def _row_key(row: pd.Series) -> tuple | None:
    try:
        date = pd.Timestamp(row.get("kickoff_utc")).date().isoformat()
        return (
            date,
            _norm(row.get("home_team")),
            _norm(row.get("away_team")),
            int(float(row.get("home_goals"))),
            int(float(row.get("away_goals"))),
        )
    except (TypeError, ValueError):
        return None

def _snapshot_keys(text: str, competition: str, season_start: int) -> set[tuple]:
    raw = text.encode("utf-8")
    frame = parse_football_txt(text, competition, int(season_start), "", raw)
    if frame.empty:
        return set()
    return {
        (
            pd.Timestamp(row.kickoff_utc).date().isoformat(),
            _norm(row.home_team),
            _norm(row.away_team),
            int(float(row.home_goals)),
            int(float(row.away_goals)),
        )
        for _, row in frame.iterrows()
    }

def _commit_time(commit: dict[str, Any]):
    raw = ((commit.get("commit") or {}).get("committer") or {}).get("date")
    return _utc(raw)

def apply_country_openfootball_pit(
    history: pd.DataFrame,
    *,
    cache_dir: str = "data/raw/pit_evidence",
    timeout: int = 30,
) -> pd.DataFrame:
    if history is None or history.empty:
        return history.copy() if history is not None else history
    out = history.copy()
    for col in ("source_available_at_utc", "pit_evidence_status", "pit_evidence_reason", "pit_evidence_url", "capture_digest"):
        if col not in out.columns:
            out[col] = pd.NA

    masks = history["competition"].astype(str).isin(SOURCE_CONFIG)
    if not masks.any():
        return out

    for (competition, season_start), group in history.loc[masks].groupby(["competition", "season_start"], sort=True):
        cfg = SOURCE_CONFIG[str(competition)]
        try:
            start_year = int(season_start)
        except (TypeError, ValueError):
            continue
        season = f"{start_year}-{str(start_year + 1)[-2:]}"
        path = cfg["file"].format(season=season)
        pending: dict[Any, tuple[tuple, Any]] = {}
        for idx, row in group.iterrows():
            if str(out.at[idx, "pit_evidence_status"]) == "VERIFIED":
                continue
            key = _row_key(row)
            bound, _reason = _result_lower_bound(row)
            if key is not None and bound is not None:
                pending[idx] = (key, bound)
        if not pending:
            continue
        try:
            commits = _commits(cfg["repo"], path, cache_dir)
        except Exception:
            continue
        ordered = []
        for commit in commits:
            dt = _commit_time(commit)
            sha = commit.get("sha")
            if dt is not None and sha:
                ordered.append((dt, str(sha)))
        ordered.sort()
        parsed: dict[str, set[tuple] | None] = {}
        for dt, sha in ordered:
            eligible = [idx for idx, (_, bound) in pending.items() if dt >= bound]
            if not eligible:
                continue
            try:
                if sha not in parsed:
                    parsed[sha] = _snapshot_keys(
                        _snapshot(cfg["repo"], path, sha, cache_dir, timeout),
                        str(competition),
                        start_year,
                    )
            except Exception:
                parsed[sha] = None
            keys = parsed[sha]
            if keys is None:
                continue
            for idx in list(eligible):
                key, bound = pending[idx]
                if dt >= bound and key in keys:
                    out.at[idx, "source_available_at_utc"] = dt.isoformat()
                    out.at[idx, "pit_evidence_status"] = "VERIFIED"
                    out.at[idx, "pit_evidence_reason"] = "immutable_openfootball_country_snapshot_after_result_lower_bound"
                    out.at[idx, "pit_evidence_url"] = f"https://github.com/{cfg['repo']}/blob/{sha}/{path}"
                    out.at[idx, "capture_digest"] = sha
                    del pending[idx]
            if not pending:
                break
    return out
