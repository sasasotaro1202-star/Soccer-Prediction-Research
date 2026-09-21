from __future__ import annotations

"""PIT evidence for openfootball's versioned Champions/Europa League text files."""

import base64
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
import pandas as pd

from src.data.openfootball_adapter import parse_football_txt

GITHUB_API = "https://api.github.com"
REPOSITORY = "openfootball/champions-league"
PATHS = {"UCL": "cl.txt", "UEL": "el.txt"}

@dataclass(frozen=True)
class OpenFootballEvidence:
    status: str
    available_at_utc: str | None = None
    evidence_url: str | None = None
    commit_sha: str | None = None
    reason: str = ""

def _utc(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _headers() -> dict[str, str]:
    h = {"Accept": "application/vnd.github+json", "User-Agent": "SoccerPredictionResearch/PIT"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

def _request(url: str, timeout: int = 30) -> requests.Response:
    r = requests.get(url, headers=_headers(), timeout=timeout)
    r.raise_for_status()
    return r

def _season(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"

def _row_key(row: pd.Series) -> tuple:
    return (
        pd.Timestamp(row["kickoff_utc"]).date().isoformat(),
        str(row["home_team"]).strip(),
        str(row["away_team"]).strip(),
        float(row["home_goals"]),
        float(row["away_goals"]),
    )

def _publication_lower_bound(row: pd.Series) -> datetime | None:
    event = _utc(row.get("kickoff_utc"))
    if event is None:
        return None
    if bool(row.get("kickoff_time_available", False)):
        return event + timedelta(minutes=180)
    return event.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)


def _snapshot_keys(text: str, competition: str, season_start: int) -> set[tuple]:
    raw = text.encode("utf-8")
    frame = parse_football_txt(text, competition, season_start, "", raw)
    if frame.empty:
        return set()
    return {_row_key(r) for _, r in frame.iterrows()}

def _commits(path: str, timeout: int) -> list[dict]:
    url = f"{GITHUB_API}/repos/{REPOSITORY}/commits?path={path}&per_page=100"
    payload = _request(url, timeout).json()
    return payload if isinstance(payload, list) else []

def _file_at_commit(path: str, sha: str, timeout: int) -> str:
    url = f"{GITHUB_API}/repos/{REPOSITORY}/contents/{path}?ref={sha}"
    payload = _request(url, timeout).json()
    if payload.get("encoding") != "base64":
        raise ValueError("unexpected GitHub contents response")
    return base64.b64decode(payload.get("content", "")).decode("utf-8", errors="replace")

def evidence_for_row(competition: str, season_start: int, row: pd.Series, timeout: int = 30) -> OpenFootballEvidence:
    if competition not in PATHS:
        return OpenFootballEvidence("UNVERIFIABLE", reason="unsupported_competition")
    event = _utc(row.get("kickoff_utc"))
    if event is None:
        return OpenFootballEvidence("UNVERIFIABLE", reason="missing_event_time")
    path = f"{_season(int(season_start))}/{PATHS[competition]}"
    try:
        commits = _commits(path, timeout)
    except Exception as exc:
        return OpenFootballEvidence("UNVERIFIABLE", reason=f"github_commit_request:{type(exc).__name__}:{exc}")
    # Publication evidence must be observed after the completed result could exist.
    # A repository commit timestamp at/before kickoff that already contains the final
    # score is not causal evidence of publication and must never be treated as PIT-safe.
    lower_bound = _publication_lower_bound(row)
    ordered = []
    for c in commits:
        dt = _utc(((c.get("commit") or {}).get("committer") or {}).get("date"))
        sha = c.get("sha")
        if dt is not None and sha and dt >= lower_bound:
            ordered.append((dt, sha))
    ordered.sort()
    wanted = _row_key(row)
    for dt, sha in ordered:
        try:
            if wanted in _snapshot_keys(_file_at_commit(path, sha, timeout), competition, int(season_start)):
                # The commit timestamp is the observed source-publication proxy.
                # Downstream PIT gates still require this timestamp to be <= the
                # prediction cutoff before the row can influence a future prediction.
                return OpenFootballEvidence(
                    "VERIFIED",
                    dt.isoformat(),
                    f"https://github.com/{REPOSITORY}/blob/{sha}/{path}",
                    sha,
                    "versioned_openfootball_snapshot_first_observed_after_result_lower_bound",
                )
        except Exception:
            continue
    return OpenFootballEvidence("UNVERIFIABLE", reason="no_versioned_snapshot_contains_completed_result")

def apply_bulk(history: pd.DataFrame, timeout: int = 30) -> pd.DataFrame:
    """Verify openfootball rows with shared versioned-snapshot caches.

    The evidence rule is unchanged: the first repository snapshot observed at or
    after the conservative result-publication lower bound must contain the exact
    completed-result identity. Shared caches avoid one GitHub API/file request per
    match, which is both slower and more likely to hit rate limits.
    """
    if history.empty:
        return history.copy()
    out = history.copy()
    for c in ("source_available_at_utc", "pit_evidence_status", "pit_evidence_reason", "pit_evidence_url", "capture_digest"):
        if c not in out.columns:
            out[c] = None
    mask = out["competition"].astype(str).isin(PATHS)
    for (competition, season_start), group in out.loc[mask].groupby(["competition", "season_start"], sort=False):
        path = f"{_season(int(season_start))}/{PATHS[str(competition)]}"
        try:
            commits = _commits(path, timeout)
        except Exception as exc:
            for idx in group.index:
                out.at[idx, "pit_evidence_status"] = "UNVERIFIABLE"
                out.at[idx, "pit_evidence_reason"] = f"github_commit_request:{type(exc).__name__}:{exc}"
            continue
        parsed_cache = {}
        ordered_commits = []
        for commit in commits:
            dt = _utc(((commit.get("commit") or {}).get("committer") or {}).get("date"))
            sha = commit.get("sha")
            if dt is not None and sha:
                ordered_commits.append((dt, sha))
        ordered_commits.sort()
        pending = {}
        for idx, row in group.iterrows():
            wanted = _row_key(row)
            lower_bound = _publication_lower_bound(row)
            if wanted is not None and lower_bound is not None:
                pending[idx] = (wanted, lower_bound)
        for dt, sha in ordered_commits:
            if not pending:
                break
            eligible = {idx for idx, (_, bound) in pending.items() if dt >= bound}
            if not eligible:
                continue
            if sha not in parsed_cache:
                try:
                    parsed_cache[sha] = _snapshot_keys(_file_at_commit(path, sha, timeout), str(competition), int(season_start))
                except Exception:
                    parsed_cache[sha] = None
            keys = parsed_cache[sha]
            if keys is None:
                continue
            for idx in list(eligible):
                wanted, bound = pending[idx]
                if wanted in keys:
                    url = f"https://github.com/{REPOSITORY}/blob/{sha}/{path}"
                    out.at[idx, "source_available_at_utc"] = dt.isoformat()
                    out.at[idx, "pit_evidence_status"] = "VERIFIED"
                    out.at[idx, "pit_evidence_reason"] = "versioned_openfootball_snapshot_first_observed_after_result_lower_bound"
                    out.at[idx, "pit_evidence_url"] = url
                    out.at[idx, "capture_digest"] = sha
                    del pending[idx]
        for idx in group.index:
            if str(out.at[idx, "pit_evidence_status"]) == "VERIFIED":
                continue
            out.at[idx, "pit_evidence_status"] = "UNVERIFIABLE"
            if pd.isna(out.at[idx, "source_available_at_utc"]):
                out.at[idx, "pit_evidence_reason"] = "no_versioned_snapshot_contains_completed_result"
    return out
