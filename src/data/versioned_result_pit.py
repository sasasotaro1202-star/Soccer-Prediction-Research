"""Bulk PIT evidence bridge using versioned openfootball/football.json snapshots.

This adapter proves only that a completed match result existed in a specific
versioned public dataset by a concrete Git commit time. It does not claim that
the original statistical source published its feature fields at that time.

Research/audit layer only. It never mutates production models or production
prediction probabilities.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

API = "https://api.github.com"
REPOSITORY = "openfootball/football.json"

COMPETITION_FILES = {
    "EPL": ("en", 1),
    "CHA": ("en", 2),
    "BL1": ("de", 1),
    "SA": ("it", 1),
    "LL": ("es", 1),
    "FL1": ("fr", 1),
    "ERE": ("nl", 1),
}

DEFAULT_PER_PAGE = 100


@dataclass(frozen=True)
class VersionedEvidence:
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


_TEAM_ALIASES = {
    "wolves": "wolverhamptonwanderers",
    "manunited": "manchesterunited",
    "manutd": "manchesterunited",
    "mancity": "manchestercity",
    "westbrom": "westbromwichalbion",
    "westham": "westhamunited",
    "spurs": "tottenhamhotspur",
    "brighton": "brightonandhovealbion",
    "newcastle": "newcastleunited",
    "leicester": "leicestercity",
    "norwich": "norwichcity",
    "stoke": "stokecity",
    "swansea": "swanseacity",
    "cardiff": "cardiffcity",
    "hull": "hullcity",
    "leeds": "leedsunited",
    "qpr": "queensparkrangers",
    "birmingham": "birminghamcity",
    "coventry": "coventrycity",
    "sheffieldutd": "sheffieldunited",
}


def _norm_team(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode(
        "ascii", "ignore"
    ).decode("ascii").casefold().strip()
    normalized = "".join(
        ch for ch in text if unicodedata.category(ch)[0] not in {"P", "Z"}
    )
    for suffix in ("fc", "afc"):
        if normalized.endswith(suffix) and len(normalized) > len(suffix) + 3:
            normalized = normalized[: -len(suffix)]
            break
    return _TEAM_ALIASES.get(normalized, normalized)


def season_path(competition: str, start_year: int) -> str:
    spec = COMPETITION_FILES.get(str(competition).upper())
    if spec is None:
        raise ValueError(f"unsupported competition: {competition}")
    language, league = spec
    return f"{int(start_year)}-{str(int(start_year) + 1)[-2:]}/{language}.{league}.json"


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "SoccerPredictionResearch/VersionedPIT",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _request(url: str, *, params: dict[str, Any] | None = None, timeout: int = 30, retries: int = 3) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, int(retries) + 1):
        try:
            response = requests.get(url, params=params, headers=_headers(), timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504}:
                retry_after = response.headers.get("Retry-After")
                if attempt < retries:
                    try:
                        delay = max(1.0, min(30.0, float(retry_after))) if retry_after else float(attempt * 2)
                    except ValueError:
                        delay = float(attempt * 2)
                    time.sleep(delay)
                    continue
            response.raise_for_status()
            return response
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(float(attempt * 2))
    raise RuntimeError(f"github request failed after {retries} attempts: {type(last_error).__name__}: {last_error}")


def _cache_key(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]


def _cache_path(cache_dir: str | Path, prefix: str, *parts: str) -> Path:
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    suffix = ".json"
    return root / f"{prefix}-{_cache_key(*parts)}{suffix}"


def _commits(path: str, *, cache_dir: str | Path, timeout: int = 30) -> list[dict[str, Any]]:
    cache = _cache_path(cache_dir, "commits", REPOSITORY, path)
    if cache.is_file() and cache.stat().st_size > 0:
        try:
            payload = json.loads(cache.read_text(encoding="utf-8"))
            if isinstance(payload, list) and payload:
                return payload
        except Exception:
            pass
    url = f"{API}/repos/{REPOSITORY}/commits"
    payload = _request(url, params={"path": path, "per_page": DEFAULT_PER_PAGE}, timeout=timeout).json()
    if not isinstance(payload, list):
        return []
    cache.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def _file_at_commit(path: str, sha: str, *, cache_dir: str | Path, timeout: int = 30) -> dict[str, Any]:
    cache = _cache_path(cache_dir, "snapshot", REPOSITORY, path, sha)
    if cache.is_file() and cache.stat().st_size > 0:
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            cache.unlink(missing_ok=True)

    url = f"{API}/repos/{REPOSITORY}/contents/{path}"
    payload = _request(url, params={"ref": sha}, timeout=timeout).json()
    if not isinstance(payload, dict) or payload.get("encoding") != "base64":
        raise ValueError("unexpected GitHub contents response")
    raw = base64.b64decode(str(payload.get("content", "")).replace("\n", ""))
    decoded = json.loads(raw.decode("utf-8"))
    cache.write_text(json.dumps(decoded, ensure_ascii=False), encoding="utf-8")
    return decoded


def _lower_bound(row: pd.Series) -> tuple[datetime | None, str]:
    kickoff = _utc(row.get("kickoff_utc"))
    if kickoff is None:
        return None, "missing_kickoff"
    if bool(row.get("kickoff_time_available", False)):
        return kickoff + timedelta(minutes=180), "kickoff_plus_180m"
    return kickoff.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1), "date_only_next_day"


def _snapshot_index(payload: dict[str, Any]) -> dict[tuple[str, str, str, float, float], int]:
    index: dict[tuple[str, str, str, float, float], int] = {}
    for match in payload.get("matches", []) if isinstance(payload, dict) else []:
        if not isinstance(match, dict):
            continue
        score = match.get("score") or {}
        ft = score.get("ft") if isinstance(score, dict) else None
        if not isinstance(ft, (list, tuple)) or len(ft) != 2:
            continue
        try:
            hg = float(ft[0])
            ag = float(ft[1])
            key = (
                str(match.get("date", ""))[:10],
                _norm_team(match.get("team1", "")),
                _norm_team(match.get("team2", "")),
                hg,
                ag,
            )
        except (TypeError, ValueError):
            continue
        index[key] = index.get(key, 0) + 1
    return index


def apply_bulk(
    history: pd.DataFrame,
    *,
    max_groups: int = 32,
    rows_per_group: int = 0,
    cache_dir: str | Path = "data/raw/versioned_pit",
    timeout: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach versioned result-publication evidence to a bounded history slice.

    max_groups and rows_per_group are explicit to keep public API use bounded.
    rows_per_group=0 means process all rows within selected groups.
    """
    if history.empty:
        return history.copy(), pd.DataFrame()

    out = history.copy()
    fields = {
        "versioned_result_source": "openfootball/football.json",
        "versioned_result_available_at_utc": None,
        "versioned_result_evidence_status": "UNVERIFIABLE",
        "versioned_result_commit_sha": None,
        "versioned_result_evidence_url": None,
        "versioned_result_evidence_reason": "",
    }
    for column, default in fields.items():
        if column not in out.columns:
            out[column] = default

    work = out.copy()
    work["competition"] = work["competition"].astype(str).str.strip().str.upper()
    work["season_start"] = pd.to_numeric(work.get("season_start"), errors="coerce")
    work["kickoff_utc"] = pd.to_datetime(work["kickoff_utc"], utc=True, errors="coerce")
    supported = work["competition"].isin(COMPETITION_FILES) & work["season_start"].notna()
    work = work.loc[supported].copy()
    groups = list(work.groupby(["competition", "season_start"], sort=True))
    if max_groups > 0:
        groups = groups[: int(max_groups)]

    report_rows: list[dict[str, Any]] = []
    for (competition, season_start), group in groups:
        rows = group
        if rows_per_group and int(rows_per_group) > 0:
            rows = group.head(int(rows_per_group))
        path = season_path(str(competition), int(season_start))
        try:
            commits = _commits(path, cache_dir=cache_dir, timeout=timeout)
        except Exception as exc:
            report_rows.append({
                "competition": str(competition),
                "season_start": int(season_start),
                "rows_considered": int(len(rows)),
                "verified_rows": 0,
                "status": "UNVERIFIABLE",
                "reason": f"commit_request:{type(exc).__name__}:{exc}",
            })
            continue

        ordered: list[tuple[datetime, str]] = []
        for commit in commits:
            dt = _utc(((commit.get("commit") or {}).get("committer") or {}).get("date"))
            sha = commit.get("sha")
            if dt is not None and sha:
                ordered.append((dt, str(sha)))
        ordered.sort()

        unresolved = set(rows.index)
        verified = 0
        ambiguous = 0

        for commit_time, sha in ordered:
            if not unresolved:
                break
            eligible_indices = []
            for idx in list(unresolved):
                row = rows.loc[idx]
                lower_bound, reason = _lower_bound(row)
                if lower_bound is not None and commit_time >= lower_bound:
                    eligible_indices.append(idx)
            if not eligible_indices:
                continue

            try:
                payload = _file_at_commit(path, sha, cache_dir=cache_dir, timeout=timeout)
                index = _snapshot_index(payload)
            except Exception:
                continue

            for idx in eligible_indices:
                row = rows.loc[idx]
                row_key = (
                    str(pd.Timestamp(row["kickoff_utc"]).date()),
                    _norm_team(row.get("home_team", "")),
                    _norm_team(row.get("away_team", "")),
                    float(row.get("home_goals")),
                    float(row.get("away_goals")),
                )
                count = int(index.get(row_key, 0))
                if count == 1:
                    out.at[idx, "versioned_result_available_at_utc"] = commit_time.isoformat()
                    out.at[idx, "versioned_result_evidence_status"] = "VERIFIED"
                    out.at[idx, "versioned_result_commit_sha"] = sha
                    out.at[idx, "versioned_result_evidence_url"] = (
                        f"https://github.com/{REPOSITORY}/blob/{sha}/{path}"
                    )
                    out.at[idx, "versioned_result_evidence_reason"] = (
                        "first_versioned_snapshot_containing_exact_completed_result_after_conservative_lower_bound"
                    )
                    unresolved.remove(idx)
                    verified += 1
                elif count > 1:
                    ambiguous += 1

        report_rows.append({
            "competition": str(competition),
            "season_start": int(season_start),
            "rows_considered": int(len(rows)),
            "verified_rows": int(verified),
            "unverifiable_rows": int(len(rows) - verified),
            "ambiguous_match_keys": int(ambiguous),
            "commit_count": int(len(ordered)),
            "status": "VERIFIED_PARTIAL" if verified else "UNVERIFIABLE",
            "source": "openfootball/football.json",
        })

    report = pd.DataFrame(report_rows)
    return out, report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="artifacts/versioned_pit_result_evidence.csv")
    parser.add_argument("--report", default="artifacts/versioned_pit_result_evidence_report.csv")
    parser.add_argument("--max-groups", type=int, default=32)
    parser.add_argument("--rows-per-group", type=int, default=0)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    enriched, report = apply_bulk(
        frame,
        max_groups=args.max_groups,
        rows_per_group=args.rows_per_group,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output, index=False)
    report.to_csv(args.report, index=False)
    print(json.dumps({
        "status": "WRITTEN",
        "rows": int(len(enriched)),
        "groups": int(len(report)),
        "verified_rows": int(
            pd.to_numeric(report.get("verified_rows", pd.Series(dtype=float)), errors="coerce").sum()
        ) if not report.empty else 0,
        "production_changed": False,
        "production_probabilities_changed": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
