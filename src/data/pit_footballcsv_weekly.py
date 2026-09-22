from __future__ import annotations

"""PIT evidence from versioned footballcsv/cache.footballdata season snapshots. 2017-18 and 2018-19 paths are retained because their immutable 2020-06-11 snapshots were independently inspected; later prediction cutoffs can safely use that observed publication proxy.

Audit-only. A match is VERIFIED only when a unique exact date/team/result
identity appears in an immutable Git snapshot whose commit timestamp is at or
after the conservative result-publication lower bound. Later snapshots may
legitimately establish availability of older results for future prediction
times; the observed commit time is never moved earlier.
"""

import hashlib
import json
import os
import time
import unicodedata
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.data.pit_source_adapter_v2 import _result_lower_bound

GITHUB_API = "https://api.github.com"
REPOSITORY = "footballcsv/cache.footballdata"

# File names confirmed in the public repository. We intentionally restrict the
# PIT provider to seasons with path-level commit histories examined in advance.
CONFIG: dict[str, dict[int, str]] = {
    "EPL": {
        2017: "2017-18/eng.1.csv",
        2018: "2018-19/eng.1.csv",
        2019: "2019-20/eng.1.csv",
        2020: "2020-21/eng.1.csv",
    },
    "BL1": {
        2017: "2017-18/de.1.csv",
        2018: "2018-19/de.1.csv",
        2019: "2019-20/de.1.csv",
        2020: "2020-21/de.1.csv",
    },
    "LL": {
        2017: "2017-18/es.1.csv",
        2018: "2018-19/es.1.csv",
        2019: "2019-20/es.1.csv",
        2020: "2020-21/es.1.csv",
    },
    "FL1": {
        2017: "2017-18/fr.1.csv",
        2018: "2018-19/fr.1.csv",
        2019: "2019-20/fr.1.csv",
        2020: "2020-21/fr.1.csv",
    },
    "SA": {
        2017: "2017-18/it.1.csv",
        2018: "2018-19/it.1.csv",
        2019: "2019-20/it.1.csv",
        2020: "2020-21/it.1.csv",
    },
    "ERE": {
        2017: "2017-18/nl.1.csv",
        2018: "2018-19/nl.1.csv",
        2019: "2019-20/nl.1.csv",
        2020: "2020-21/nl.1.csv",
    },
}


def _utc(value: Any) -> pd.Timestamp | None:
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
    return stamp.tz_convert("UTC")


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "SoccerPredictionResearch/footballcsv-weekly-PIT",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _request_json(url: str, timeout: int = 30) -> Any:
    last_error: Exception | None = None
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
            last_error = exc
            if attempt >= 3:
                break
            time.sleep(float(attempt * 2))
    raise RuntimeError(
        f"github_request_failed:{type(last_error).__name__}:{last_error}"
        if last_error is not None
        else "github_request_failed"
    )


def _cache_file(cache_dir: str, prefix: str, *parts: str) -> Path:
    key = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]
    suffix = ".json" if prefix == "commits" else ".csv"
    path = Path(cache_dir) / f"footballcsv-weekly-{prefix}-{key}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _commit_time(commit: dict[str, Any]) -> pd.Timestamp | None:
    return _utc(((commit.get("commit") or {}).get("committer") or {}).get("date"))


def _commits(path: str, cache_dir: str, timeout: int) -> list[dict[str, Any]]:
    cache = _cache_file(cache_dir, "commits", REPOSITORY, path)
    try:
        refresh_hours = float(
            os.getenv("PIT_FOOTBALLCSV_WEEKLY_COMMIT_CACHE_REFRESH_HOURS", "6")
        )
    except ValueError:
        refresh_hours = 6.0

    if cache.exists() and refresh_hours > 0:
        try:
            if (time.time() - cache.stat().st_mtime) <= refresh_hours * 3600.0:
                payload = json.loads(cache.read_text(encoding="utf-8"))
                if isinstance(payload, list) and payload:
                    return payload
        except Exception:
            pass

    payload = _request_json(
        f"{GITHUB_API}/repos/{REPOSITORY}/commits"
        f"?path={path}&per_page=100&page=1",
        timeout,
    )
    if not isinstance(payload, list):
        raise RuntimeError("invalid_footballcsv_weekly_commit_history")
    if payload:
        cache.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def _snapshot(path: str, sha: str, cache_dir: str, timeout: int) -> str:
    cache = _cache_file(cache_dir, "snapshot", REPOSITORY, path, sha)
    if cache.exists():
        return cache.read_text(encoding="utf-8")

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = requests.get(
                f"https://raw.githubusercontent.com/{REPOSITORY}/{sha}/{path}",
                headers={"User-Agent": "SoccerPredictionResearch/footballcsv-weekly-PIT"},
                timeout=timeout,
            )
            response.raise_for_status()
            text = response.content.decode("utf-8", errors="replace")
            if text.lstrip().lower().startswith(("<!doctype html", "<html")):
                raise ValueError("html_instead_of_snapshot")
            cache.write_text(text, encoding="utf-8")
            return text
        except (requests.RequestException, OSError, ValueError) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(float(attempt * 2))

    raise RuntimeError(
        f"snapshot_fetch_failed:{type(last_error).__name__}:{last_error}"
        if last_error is not None
        else "snapshot_fetch_failed"
    )

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
    frame = pd.read_csv(StringIO(text), low_memory=False)
    required = {"Date", "Team 1", "FT", "Team 2"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"footballcsv weekly snapshot missing columns: {sorted(required - set(frame.columns))}"
        )
    counts: dict[tuple, int] = {}
    for _, row in frame.iterrows():
        try:
            date = pd.Timestamp(row["Date"]).date().isoformat()
            home = _norm(row["Team 1"])
            away = _norm(row["Team 2"])
            ft = str(row["FT"] or "").strip()
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


def apply_footballcsv_weekly_pit(
    history: pd.DataFrame,
    *,
    cache_dir: str = "data/raw/pit_evidence",
    timeout: int = 30,
) -> pd.DataFrame:
    if history is None or history.empty:
        return history.copy() if history is not None else history

    verified: list[dict[str, Any]] = []
    for competition, season_paths in CONFIG.items():
        mask = history["competition"].astype(str).eq(competition)
        if not mask.any():
            continue

        for season_start, group in history.loc[mask].groupby("season_start", sort=True):
            try:
                year = int(season_start)
            except (TypeError, ValueError):
                continue
            path = season_paths.get(year)
            if not path:
                continue

            pending: dict[Any, tuple[tuple, pd.Timestamp]] = {}
            for idx, row in group.iterrows():
                if str(row.get("pit_evidence_status", "")) == "VERIFIED":
                    continue
                key = _target_key(row)
                lower_bound, _reason = _result_lower_bound(row)
                lower_bound_ts = _utc(lower_bound)
                if key is not None and lower_bound_ts is not None:
                    pending[idx] = (key, lower_bound_ts)
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
            for observed_at, sha in ordered:
                if not pending:
                    break
                eligible = [
                    idx
                    for idx, (_, lower_bound) in pending.items()
                    if observed_at >= lower_bound
                ]
                if not eligible:
                    continue

                if sha not in parsed_cache:
                    try:
                        parsed_cache[sha] = _source_rows(
                            _snapshot(path, sha, cache_dir, timeout)
                        )
                    except Exception:
                        parsed_cache[sha] = None
                counts = parsed_cache[sha]
                if counts is None:
                    continue

                for idx in list(eligible):
                    key, _lower_bound = pending[idx]
                    # Unique exact identity only. Ambiguous duplicate source rows
                    # are deliberately not evidence.
                    if counts.get(key, 0) != 1:
                        continue
                    verified.append(
                        {
                            "source_available_at_utc": observed_at.isoformat(),
                            "pit_evidence_status": "VERIFIED",
                            "pit_evidence_reason": (
                                "immutable_footballcsv_weekly_snapshot_contains_unique_"
                                "exact_result_after_result_lower_bound"
                            ),
                            "pit_evidence_url": (
                                f"https://github.com/{REPOSITORY}/blob/{sha}/{path}"
                            ),
                            "capture_digest": sha,
                            "__index__": idx,
                        }
                    )
                    del pending[idx]

    if not verified:
        return history.iloc[0:0].copy()
    out = pd.DataFrame(verified).set_index("__index__")
    out.index.name = history.index.name
    return out
