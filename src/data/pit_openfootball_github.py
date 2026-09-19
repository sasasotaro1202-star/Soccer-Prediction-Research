from __future__ import annotations

"""PIT evidence for openfootball's versioned Champions/Europa League text files."""

import base64
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
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
    ordered = []
    for c in commits:
        dt = _utc(((c.get("commit") or {}).get("committer") or {}).get("date"))
        sha = c.get("sha")
        if dt is not None and sha and dt <= event:
            ordered.append((dt, sha))
    ordered.sort()
    wanted = _row_key(row)
    for dt, sha in ordered:
        try:
            if wanted in _snapshot_keys(_file_at_commit(path, sha, timeout), competition, int(season_start)):
                # A versioned result snapshot that already contains the final score
                # is not proof of pre-match availability. Only snapshots at/before
                # kickoff can be considered PIT candidates; downstream gates still
                # require an explicit pre-cutoff evidence record.
                return OpenFootballEvidence(
                    "VERIFIED",
                    dt.isoformat(),
                    f"https://github.com/{REPOSITORY}/blob/{sha}/{path}",
                    sha,
                    "versioned_openfootball_snapshot_contains_result_at_or_before_kickoff",
                )
        except Exception:
            continue
    return OpenFootballEvidence("UNVERIFIABLE", reason="no_versioned_snapshot_contains_completed_result")

def apply_bulk(history: pd.DataFrame, timeout: int = 30) -> pd.DataFrame:
    if history.empty:
        return history.copy()
    out = history.copy()
    for c in ("source_available_at_utc", "pit_evidence_status", "pit_evidence_reason", "pit_evidence_url", "capture_digest"):
        if c not in out.columns:
            out[c] = None
    mask = out["competition"].astype(str).isin(PATHS)
    for idx, row in out.loc[mask].iterrows():
        ev = evidence_for_row(str(row["competition"]), int(row["season_start"]), row, timeout)
        out.at[idx, "source_available_at_utc"] = ev.available_at_utc
        out.at[idx, "pit_evidence_status"] = ev.status
        out.at[idx, "pit_evidence_reason"] = ev.reason
        out.at[idx, "pit_evidence_url"] = ev.evidence_url
        out.at[idx, "capture_digest"] = ev.commit_sha
    return out
