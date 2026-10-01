from __future__ import annotations

"""Conservative ingestion of discovered public GitHub match-result datasets.

A discovered repository is never trusted merely because search found it. A file
must expose explicit match date/team/score fields, have a plausible competition
match, and have an immutable Git commit timestamp recorded as the conservative
source-availability timestamp. Rows then enter the normal PIT gate.
"""

import base64
import hashlib
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests


GITHUB_API = "https://api.github.com"
TARGET_ALIASES = {
    "EPL": ("epl", "premier league", "england premier", "e0"),
    "BL1": ("bl1", "bundesliga", "germany 1", "d1"),
    "SA": ("sa", "serie a", "italy 1", "i1"),
    "LL": ("ll", "la liga", "laliga", "spain 1", "sp1"),
    "FL1": ("fl1", "ligue 1", "france 1", "f1"),
    "ERE": ("ere", "eredivisie", "netherlands 1", "n1"),
    "J1": ("j1", "j.league", "j league"),
    "J2": ("j2", "j2 league"),
    "J3": ("j3", "j3 league"),
    "UCL": ("ucl", "champions league", "uefa champions"),
    "UEL": ("uel", "europa league", "uefa europa"),
    "UECL": ("uecl", "conference league", "uefa conference"),
    "WORLD_CUP": ("world cup", "worldcup", "fifa world cup"),
    "ASIAN_CUP": ("asian cup", "afc asian cup", "asiancup"),
}


def _api_get(url: str, *, timeout: int = 20) -> requests.Response:
    response = requests.get(
        url,
        timeout=timeout,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "SoccerPredictionResearch-candidate-adapter",
        },
    )
    response.raise_for_status()
    return response


def _norm(s: Any) -> str:
    return (
        str(s)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def competition_path_matches(competition: str, path: str) -> bool:
    comp = str(competition).strip().upper()
    aliases = TARGET_ALIASES.get(comp, (comp.lower(),))
    text = _norm(path)
    return any(alias in text for alias in aliases)


def _pick_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {_norm(c): c for c in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for original in columns:
        n = _norm(original)
        if any(candidate in n for candidate in candidates):
            return original
    return None


def infer_match_columns(columns: list[str]) -> dict[str, str | None]:
    return {
        "date": _pick_column(
            columns,
            ("date", "match date", "match_date", "event date", "event_date", "kickoff", "datetime"),
        ),
        "home_team": _pick_column(
            columns,
            ("home team", "home_team", "hometeam", "home name", "home"),
        ),
        "away_team": _pick_column(
            columns,
            ("away team", "away_team", "awayteam", "away name", "away"),
        ),
        "home_goals": _pick_column(
            columns,
            ("home goals", "home_goals", "fthg", "home score", "home_score", "goals home"),
        ),
        "away_goals": _pick_column(
            columns,
            ("away goals", "away_goals", "ftag", "away score", "away_score", "goals away"),
        ),
        "competition": _pick_column(
            columns,
            ("competition", "league", "tournament", "competition name", "league name"),
        ),
        "season": _pick_column(columns, ("season", "season start", "season_start", "year")),
        "result": _pick_column(columns, ("result", "ftr", "outcome")),
    }


def _latest_path_commit(full_name: str, path: str) -> datetime:
    encoded = quote(path, safe="")
    payload = _api_get(
        f"{GITHUB_API}/repos/{full_name}/commits?path={encoded}&per_page=1"
    ).json()
    if not payload:
        raise ValueError("No commit history for candidate file")
    timestamp = payload[0].get("commit", {}).get("committer", {}).get("date")
    parsed = pd.to_datetime(timestamp, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError("Candidate file commit timestamp is missing")
    return parsed.to_pydatetime()


def _candidate_files(full_name: str, branch: str, *, max_files: int = 8) -> list[str]:
    encoded_branch = quote(str(branch), safe="")
    payload = _api_get(
        f"{GITHUB_API}/repos/{full_name}/git/trees/{encoded_branch}?recursive=1"
    ).json()
    if payload.get("truncated") is True:
        raise ValueError("GitHub candidate tree is truncated")
    paths = [
        str(item.get("path", ""))
        for item in payload.get("tree", [])
        if item.get("type") == "blob"
        and str(item.get("path", "")).lower().endswith(".csv")
    ]
    bad = ("readme", "docs/", "test/", "tests/", "notebook", "node_modules/")
    paths = [p for p in paths if not any(token in p.lower() for token in bad)]
    paths.sort(
        key=lambda p: (
            0
            if any(k in p.lower() for k in ("results", "matches", "fixtures", "games", "football"))
            else 1,
            len(p),
        )
    )
    return paths[: int(max_files)]


def _fetch_csv(full_name: str, branch: str, path: str, *, max_bytes: int = 2_000_000) -> bytes:
    encoded = quote(path, safe="/")
    payload = _api_get(
        f"{GITHUB_API}/repos/{full_name}/contents/{encoded}?ref={quote(str(branch), safe='')}"
    ).json()
    size = int(payload.get("size", 0) or 0)
    if size <= 0 or size > int(max_bytes):
        raise ValueError(f"Candidate CSV size {size} exceeds safe limit")
    encoded_content = payload.get("content")
    if not isinstance(encoded_content, str):
        raise ValueError("Candidate CSV content is not available through GitHub API")
    return base64.b64decode(encoded_content.replace("\n", ""))


def adapt_candidate_csv(
    raw: bytes,
    *,
    competition: str,
    source_repo: str,
    source_path: str,
    source_available_at_utc: datetime,
) -> pd.DataFrame:
    sample = pd.read_csv(BytesIO(raw), nrows=5000)
    columns = infer_match_columns(list(sample.columns))
    required = ("date", "home_team", "away_team", "home_goals", "away_goals")
    if any(columns[name] is None for name in required):
        raise ValueError("Candidate CSV lacks explicit date/team/score columns")

    comp_col = columns.get("competition")
    if comp_col is None and not competition_path_matches(competition, source_path):
        raise ValueError("Competition is not explicit and file path does not identify target competition")

    date = pd.to_datetime(sample[columns["date"]], utc=True, errors="coerce")
    home = sample[columns["home_team"]].astype("string").str.strip()
    away = sample[columns["away_team"]].astype("string").str.strip()
    hg = pd.to_numeric(sample[columns["home_goals"]], errors="coerce")
    ag = pd.to_numeric(sample[columns["away_goals"]], errors="coerce")
    mask = date.notna() & home.notna() & away.notna() & hg.notna() & ag.notna()
    mask &= hg >= 0
    mask &= ag >= 0
    if comp_col is not None:
        comp_text = sample[comp_col].astype("string").map(_norm)
        aliases = TARGET_ALIASES.get(competition, (str(competition).lower(),))
        mask &= comp_text.apply(lambda value: any(alias in str(value) for alias in aliases))

    if not mask.any():
        raise ValueError("Candidate CSV contains no validated rows for target competition")

    out = pd.DataFrame({
        "competition": str(competition),
        "season": date.dt.year.astype("Int64").astype("string"),
        "season_start": date.dt.year,
        "kickoff_utc": date,
        "kickoff_time_available": date.dt.hour.ne(0) | date.dt.minute.ne(0),
        "event_time_precision": "MINUTE",
        "home_team": home,
        "away_team": away,
        "home_goals": hg,
        "away_goals": ag,
        "result": sample[columns["result"]].astype("string").str.strip() if columns.get("result") else pd.Series(pd.NA, index=sample.index, dtype="string"),
        "source_name": f"GitHub:{source_repo}",
        "source_record_id": [str(i) for i in sample.index],
        "source_path": source_path,
        "source_available_at_utc": pd.Timestamp(source_available_at_utc),
        "retrieved_at_utc": datetime.now(timezone.utc),
    }, index=sample.index)
    out["event_time_precision"] = out["event_time_precision"].mask(~out["kickoff_time_available"], "DATE_ONLY")
    out["match_id"] = [
        "gh:"
        + hashlib.sha256(
            f"{source_repo}|{source_path}|{idx}|{row['kickoff_utc']}|{row['home_team']}|{row['away_team']}".encode()
        ).hexdigest()[:24]
        for idx, row in out.iterrows()
    ]
    return out.loc[mask].reset_index(drop=True)


def acquire_discovered_candidates(
    discovery: dict[str, Any],
    *,
    max_competitions: int = 4,
    max_candidates_total: int = 6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    acquired: list[pd.DataFrame] = []
    report: dict[str, Any] = {
        "status": "COMPLETED",
        "max_candidates_total": int(max_candidates_total),
        "candidates": [],
    }
    count = 0
    for competition, candidates in (discovery.get("competitions") or {}).items():
        if count >= int(max_candidates_total):
            break
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if count >= int(max_candidates_total):
                break
            if not isinstance(candidate, dict):
                continue
            full_name = str(candidate.get("full_name", "")).strip()
            branch = str(candidate.get("default_branch", "main")).strip() or "main"
            if not full_name:
                continue
            item = {
                "competition": str(competition),
                "repository": full_name,
                "status": "REJECTED",
            }
            try:
                paths = _candidate_files(full_name, branch)
                used = False
                for path in paths:
                    try:
                        commit_time = _latest_path_commit(full_name, path)
                        raw = _fetch_csv(full_name, branch, path)
                        frame = adapt_candidate_csv(
                            raw,
                            competition=str(competition),
                            source_repo=full_name,
                            source_path=path,
                            source_available_at_utc=commit_time,
                        )
                        if not frame.empty:
                            acquired.append(frame)
                            item.update({
                                "status": "ACQUIRED_RESEARCH",
                                "path": path,
                                "rows": int(len(frame)),
                                "source_available_at_utc": commit_time.isoformat(),
                            })
                            used = True
                            count += 1
                            break
                    except Exception:
                        continue
                if not used:
                    item["reason"] = "no_candidate_file_passed_schema_and_target_checks"
            except Exception as exc:
                item["reason"] = f"{type(exc).__name__}: {exc}"
            report["candidates"].append(item)
    history = (
        pd.concat(acquired, ignore_index=True)
        if acquired else pd.DataFrame()
    )
    report["acquired_rows"] = int(len(history))
    report["acquired_candidates"] = int(sum(x.get("status") == "ACQUIRED_RESEARCH" for x in report["candidates"]))
    if report["acquired_candidates"] == 0 and report["candidates"]:
        report["status"] = "NO_CANDIDATE_PASSED"
    return history, report
