from __future__ import annotations

"""Versioned PIT evidence from openfootball/football.json.

This is an evidence-recovery provider, not a prediction-data source.  A match is
verified only when a versioned repository snapshot, committed at/after a
conservative result-availability lower bound, contains the exact completed
result.  Commit time is treated as an observed publication proxy and is never
moved earlier than the commit timestamp.
"""

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_URL = "https://github.com/openfootball/football.json.git"
REPO_DIRNAME = "openfootball-football-json"
COMPETITION_TO_FILE = {
    "EPL": "en.1.json",
    "CHA": "en.2.json",
    "BL1": "de.1.json",
    "SA": "it.1.json",
    "LL": "es.1.json",
    "FL1": "fr.1.json",
    "ERE": "nl.1.json",
}
INPUT_TO_CANONICAL = {"E0": "EPL", "CH": "CHA", "D1": "BL1", "I1": "SA", "SP1": "LL", "F1": "FL1", "N1": "ERE"}


def _utc(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _lower_bound(row: pd.Series) -> datetime | None:
    event = _utc(row.get("kickoff_utc"))
    if event is None:
        return None
    if bool(row.get("kickoff_time_available", False)):
        return event + timedelta(minutes=180)
    return event.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)


def _norm(value: object) -> str:
    return "".join(ch for ch in str(value).casefold() if ch.isalnum())


def _row_key(row: pd.Series) -> tuple | None:
    event = _utc(row.get("kickoff_utc"))
    if event is None:
        return None
    try:
        return (
            event.date().isoformat(),
            _norm(row.get("home_team", "")),
            _norm(row.get("away_team", "")),
            int(float(row.get("home_goals"))),
            int(float(row.get("away_goals"))),
        )
    except (TypeError, ValueError):
        return None


def _snapshot_keys(raw: str) -> set[tuple]:
    payload = json.loads(raw)
    keys = set()
    for match in payload.get("matches", []):
        score = match.get("score")
        if isinstance(score, dict):
            ft = score.get("ft")
        else:
            ft = score
        if not isinstance(ft, (list, tuple)) or len(ft) != 2:
            continue
        try:
            date = str(match.get("date", ""))[:10]
            if not date:
                continue
            keys.add((date, _norm(match.get("team1", "")), _norm(match.get("team2", "")), int(ft[0]), int(ft[1])))
        except (TypeError, ValueError):
            continue
    return keys


def _run(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)
    return result.stdout


def _ensure_repo(cache_dir: Path) -> Path:
    repo = cache_dir / REPO_DIRNAME
    if (repo / ".git").is_dir():
        return repo
    cache_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", REPO_URL, str(repo)], check=True, capture_output=True, text=True)
    return repo


def _season_path(season_start: int, filename: str) -> str:
    return f"{int(season_start)}-{str(int(season_start) + 1)[-2:]}/{filename}"


def _commit_snapshots(repo: Path, path: str, min_time: datetime) -> list[tuple[datetime, str]]:
    raw = _run(repo, "log", "--format=%H%x09%cI", "--reverse", "--follow", "--", path)
    rows = []
    for line in raw.splitlines():
        sha, stamp = line.split("\t", 1)
        dt = _utc(stamp)
        if dt is not None and dt >= min_time:
            rows.append((dt, sha))
    return rows


def apply_openfootball_history(history: pd.DataFrame, cache_dir: str = "data/raw/pit_evidence") -> pd.DataFrame:
    if history is None or history.empty:
        return history.copy() if history is not None else history
    out = history.copy()
    for c in ("source_available_at_utc", "pit_evidence_status", "pit_evidence_reason", "pit_evidence_url", "capture_digest"):
        if c not in out.columns:
            out[c] = pd.NA

    pending_groups = {}
    for idx, row in out.iterrows():
        if str(row.get("pit_evidence_status", "")) == "VERIFIED":
            continue
        canonical = INPUT_TO_CANONICAL.get(str(row.get("competition")))
        if canonical not in COMPETITION_TO_FILE:
            continue
        key = _row_key(row)
        bound = _lower_bound(row)
        if key is None or bound is None:
            continue
        try:
            season_start = int(row.get("season_start"))
        except (TypeError, ValueError):
            continue
        path = _season_path(season_start, COMPETITION_TO_FILE[canonical])
        pending_groups.setdefault(path, {})[idx] = (key, bound, canonical, season_start)

    if not pending_groups:
        return out

    repo = _ensure_repo(Path(cache_dir))
    for path, pending in pending_groups.items():
        try:
            first_bound = min(item[1] for item in pending.values())
            commits = _commit_snapshots(repo, path, first_bound)
        except Exception as exc:
            for idx in pending:
                out.at[idx, "pit_evidence_status"] = "UNVERIFIABLE"
                out.at[idx, "pit_evidence_reason"] = f"openfootball_history_error:{type(exc).__name__}:{exc}"
            continue

        for dt, sha in commits:
            if not pending:
                break
            try:
                raw = _run(repo, "show", f"{sha}:{path}")
                keys = _snapshot_keys(raw)
            except Exception:
                continue
            for idx in list(pending):
                key, bound, canonical, season_start = pending[idx]
                if dt >= bound and key in keys:
                    out.at[idx, "source_available_at_utc"] = dt.isoformat()
                    out.at[idx, "pit_evidence_status"] = "VERIFIED"
                    out.at[idx, "pit_evidence_reason"] = "versioned_openfootball_json_snapshot_after_result_lower_bound"
                    out.at[idx, "pit_evidence_url"] = f"https://github.com/openfootball/football.json/blob/{sha}/{path}"
                    out.at[idx, "capture_digest"] = sha
                    del pending[idx]
        for idx in pending:
            if str(out.at[idx, "pit_evidence_status"]) != "VERIFIED":
                out.at[idx, "pit_evidence_status"] = "UNVERIFIABLE"
                out.at[idx, "pit_evidence_reason"] = "no_versioned_openfootball_snapshot_contains_completed_result"
    return out
