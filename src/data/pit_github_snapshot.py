from __future__ import annotations

"""Point-in-time evidence from versioned public football datasets on GitHub.

This adapter treats a Git commit as the publication time of that dataset version.
A row is VERIFIED only when the completed result is present in a committed snapshot
whose commit timestamp is after the conservative result-availability lower bound.
The adapter never treats the current HEAD timestamp as historical publication time.
"""

import base64
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests

GITHUB_API = "https://api.github.com"
REPOSITORY = "openfootball/football.json"

# Competition -> (season directory, dataset filename).  Only competitions with a
# stable openfootball representation are enabled here; all others remain unknown.
DATASET_FILES = {
    "EPL": ("en", 1),
    "CHA": ("en", 2),
    "BL1": ("de", 1),
    "SA": ("it", 1),
    "LL": ("es", 1),
    "FL1": ("fr", 1),
}


@dataclass(frozen=True)
class GitHubSnapshotEvidence:
    status: str
    available_at_utc: str | None = None
    commit_sha: str | None = None
    evidence_url: str | None = None
    reason: str = ""


def _utc(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    text = str(value).strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _season_dir(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def dataset_path(competition: str, start_year: int) -> str:
    spec = DATASET_FILES.get(str(competition))
    if spec is None:
        raise ValueError(f"No versioned GitHub dataset mapping for {competition}")
    language, league = spec
    return f"{_season_dir(int(start_year))}/{language}.{league}.json"


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "SoccerPredictionResearch/PIT"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _request(url: str, timeout: int = 30) -> requests.Response:
    response = requests.get(url, headers=_headers(), timeout=timeout)
    response.raise_for_status()
    return response


def _row_key(date: str, home: str, away: str, home_goals: float, away_goals: float) -> tuple:
    return (str(date)[:10], str(home).strip(), str(away).strip(), float(home_goals), float(away_goals))


def _snapshot_keys(payload: dict) -> set[tuple]:
    keys: set[tuple] = set()
    for match in payload.get("matches", []) if isinstance(payload, dict) else []:
        score = match.get("score") or {}
        ft = score.get("ft") if isinstance(score, dict) else None
        if not isinstance(ft, (list, tuple)) or len(ft) != 2:
            continue
        try:
            keys.add(_row_key(match.get("date"), match.get("team1"), match.get("team2"), ft[0], ft[1]))
        except (TypeError, ValueError):
            continue
    return keys


def _commits(path: str, timeout: int = 30) -> list[dict]:
    url = f"{GITHUB_API}/repos/{REPOSITORY}/commits?path={path}&per_page=100"
    payload = _request(url, timeout=timeout).json()
    return payload if isinstance(payload, list) else []


def _file_at_commit(path: str, sha: str, timeout: int = 30) -> dict:
    url = f"{GITHUB_API}/repos/{REPOSITORY}/contents/{path}?ref={sha}"
    payload = _request(url, timeout=timeout).json()
    if not isinstance(payload, dict) or payload.get("encoding") != "base64":
        raise ValueError("unexpected GitHub contents response")
    raw = base64.b64decode(payload.get("content", ""))
    import json
    return json.loads(raw.decode("utf-8"))


def evidence_for_row(
    competition: str,
    start_year: int,
    event_time_utc: str,
    home_team: str,
    away_team: str,
    home_goals: float,
    away_goals: float,
    timeout: int = 30,
) -> GitHubSnapshotEvidence:
    """Return the first versioned snapshot proving the completed result existed.

    The commit must be strictly after result availability.  Because the source is
    versioned, this is valid PIT evidence for this source itself; it does not claim
    that Football-Data.co.uk had published the row at the same time.
    """
    lower_bound = _utc(event_time_utc)
    if lower_bound is None:
        return GitHubSnapshotEvidence("UNVERIFIABLE", reason="missing_event_time")
    try:
        path = dataset_path(competition, start_year)
    except ValueError as exc:
        return GitHubSnapshotEvidence("UNVERIFIABLE", reason=str(exc))

    try:
        commits = _commits(path, timeout=timeout)
    except requests.RequestException as exc:
        return GitHubSnapshotEvidence("UNVERIFIABLE", reason=f"github_commit_request:{type(exc).__name__}:{exc}")
    except ValueError as exc:
        return GitHubSnapshotEvidence("UNVERIFIABLE", reason=f"github_commit_parse:{exc}")

    ordered = []
    for commit in commits:
        dt = _utc(((commit.get("commit") or {}).get("committer") or {}).get("date"))
        sha = commit.get("sha")
        if dt is not None and sha and dt >= lower_bound:
            ordered.append((dt, sha))
    ordered.sort()
    if not ordered:
        return GitHubSnapshotEvidence("UNVERIFIABLE", reason="no_commit_after_result_lower_bound")

    wanted = _row_key(event_time_utc, home_team, away_team, home_goals, away_goals)
    for commit_time, sha in ordered:
        try:
            payload = _file_at_commit(path, sha, timeout=timeout)
            if wanted in _snapshot_keys(payload):
                return GitHubSnapshotEvidence(
                    "VERIFIED",
                    available_at_utc=commit_time.isoformat(),
                    commit_sha=sha,
                    evidence_url=f"https://github.com/{REPOSITORY}/blob/{sha}/{path}",
                    reason="versioned_dataset_snapshot_contains_completed_result",
                )
        except (requests.RequestException, ValueError, UnicodeDecodeError):
            continue
    return GitHubSnapshotEvidence("UNVERIFIABLE", reason="no_versioned_snapshot_contains_completed_result")
