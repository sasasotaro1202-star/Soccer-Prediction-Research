from __future__ import annotations

"""PIT evidence from the versioned J1 2020 match dataset.

The provider is audit-only. A canonical fixture is marked VERIFIED only when:
1) a source commit is at/after the conservative result-availability lower bound,
2) that immutable snapshot contains the exact home/away/full-time result, and
3) the snapshot contains that identity exactly once.

The source repository is fixed to a public historical J1 dataset whose file was
updated during the 2020 season. No current HEAD timestamp is treated as historical
publication time.
"""

import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import time
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.data.pit_source_adapter_v2 import _result_lower_bound

GITHUB_API = "https://api.github.com"
REPOSITORY = "ewalldo/Japan-J1-League-Data-and-Data-Analysis"
BRANCH = "master"
PATH = "Match Data/JLeague-2020.csv"
COMPETITION = "J1"
SEASON_START = 2020


def _utc(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "SoccerPredictionResearch/JLeague2020PIT",
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
                delay = float(retry_after) if retry_after and str(retry_after).replace(".", "", 1).isdigit() else float(attempt * 2)
                time.sleep(min(30.0, max(1.0, delay)))
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt >= 3:
                raise
            time.sleep(float(attempt * 2))
    raise RuntimeError("github_request_failed_after_retries")


def _cache_path(cache_dir: str, prefix: str, key: str, suffix: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    path = Path(cache_dir) / f"jleague2020-{prefix}-{digest}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _commit_time(commit: dict[str, Any]) -> pd.Timestamp | None:
    return _utc(((commit.get("commit") or {}).get("committer") or {}).get("date"))


def _git_repo_dir(cache_dir: str) -> Path:
    return Path(cache_dir) / "jleague2020-git-repository"


def _ensure_git_repo(cache_dir: str, *, timeout: int = 120) -> Path:
    if shutil.which("git") is None:
        raise RuntimeError("git_binary_unavailable_for_jleague2020_pit_fallback")
    repo_dir = _git_repo_dir(cache_dir)
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if (repo_dir / ".git").is_dir():
        subprocess.run(
            ["git", "-C", str(repo_dir), "fetch", "--quiet", "--no-tags", "origin", BRANCH],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
        )
        return repo_dir
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    subprocess.run(
        [
            "git", "clone", "--filter=blob:none", "--no-checkout",
            "--single-branch", "--branch", BRANCH,
            f"https://github.com/{REPOSITORY}.git", str(repo_dir),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        text=True,
    )
    return repo_dir


def _parse_git_log(output: str) -> list[dict[str, Any]]:
    commits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in output.splitlines():
        sha, sep, committed_at = raw.strip().partition("\t")
        sha = sha.strip()
        committed_at = committed_at.strip()
        if not sep or len(sha) < 40 or not committed_at or sha in seen:
            continue
        seen.add(sha)
        commits.append({
            "sha": sha,
            "commit": {"committer": {"date": committed_at}},
        })
    return commits


def _git_commits(cache_dir: str, *, timeout: int = 120) -> list[dict[str, Any]]:
    repo_dir = _ensure_git_repo(cache_dir, timeout=timeout)
    completed = subprocess.run(
        ["git", "-C", str(repo_dir), "log", "--format=%H%x09%cI", "--all", "--", PATH],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        text=True,
    )
    commits = _parse_git_log(completed.stdout)
    if not commits:
        raise RuntimeError("jleague2020_git_history_empty")
    return commits


def _commits(cache_dir: str, *, timeout: int = 30) -> list[dict[str, Any]]:
    cache = _cache_path(cache_dir, "commits", REPOSITORY + "|" + PATH, ".json")
    if cache.exists():
        try:
            payload = json.loads(cache.read_text(encoding="utf-8"))
            if isinstance(payload, list) and payload:
                return payload
        except Exception:
            pass

    try:
        payload = _request_json(
            f"{GITHUB_API}/repos/{REPOSITORY}/commits"
            f"?path={requests.utils.quote(PATH)}&per_page=100&page=1",
            timeout=timeout,
        )
        if not isinstance(payload, list):
            raise RuntimeError("jleague2020_commit_history_invalid_response")
        commits = [x for x in payload if isinstance(x, dict) and x.get("sha")]
        if commits:
            cache.write_text(json.dumps(commits, ensure_ascii=False), encoding="utf-8")
        return commits
    except Exception as api_exc:
        # Git transport is independent of the REST API quota. Prefer it as a
        # free, immutable-history recovery path before declaring PIT evidence
        # unavailable. The verification semantics are unchanged: commit SHA,
        # committer timestamp and exact snapshot contents are still required.
        try:
            commits = _git_commits(cache_dir, timeout=max(60, timeout))
            cache.write_text(json.dumps(commits, ensure_ascii=False), encoding="utf-8")
            return commits
        except Exception as git_exc:
            raise RuntimeError(
                f"jleague2020_commit_history_unavailable:api={type(api_exc).__name__}:{api_exc};"
                f"git={type(git_exc).__name__}:{git_exc}"
            ) from git_exc


def _git_snapshot_text(sha: str, cache_dir: str, *, timeout: int = 120) -> str:
    repo_dir = _ensure_git_repo(cache_dir, timeout=timeout)
    completed = subprocess.run(
        ["git", "-C", str(repo_dir), "show", f"{sha}:{PATH}"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    text = completed.stdout.decode("utf-8-sig", errors="replace")
    if text.lstrip().lower().startswith(("<!doctype html", "<html")):
        raise ValueError("html_instead_of_jleague_snapshot")
    return text


def _snapshot_text(sha: str, cache_dir: str, *, timeout: int = 30) -> str:
    cache = _cache_path(cache_dir, "snapshot", sha, ".csv")
    if cache.exists():
        return cache.read_text(encoding="utf-8-sig")

    url = (
        f"https://raw.githubusercontent.com/{REPOSITORY}/{sha}/"
        f"Match%20Data/JLeague-2020.csv"
    )
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = requests.get(
                url,
                headers={"User-Agent": "SoccerPredictionResearch/JLeague2020PIT"},
                timeout=timeout,
            )
            response.raise_for_status()
            text = response.content.decode("utf-8-sig", errors="replace")
            if text.lstrip().lower().startswith(("<!doctype html", "<html")):
                raise ValueError("html_instead_of_jleague_snapshot")
            cache.write_text(text, encoding="utf-8")
            return text
        except (requests.RequestException, OSError, ValueError) as exc:
            last_exc = exc
            if attempt < 3:
                time.sleep(float(attempt * 2))

    # Raw GitHub is outside the REST API path, but a local immutable clone is
    # the final free recovery path when raw serving itself is unavailable.
    try:
        text = _git_snapshot_text(sha, cache_dir, timeout=max(60, timeout))
        cache.write_text(text, encoding="utf-8")
        return text
    except Exception as git_exc:
        if last_exc is not None:
            raise RuntimeError(
                f"jleague2020_snapshot_fetch_failed:raw={type(last_exc).__name__}:{last_exc};"
                f"git={type(git_exc).__name__}:{git_exc}"
            ) from git_exc
        raise RuntimeError(f"jleague2020_snapshot_fetch_failed:git={type(git_exc).__name__}:{git_exc}") from git_exc


def _norm(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    return "".join(ch.casefold() for ch in text if ch.isalnum())


def _source_rows(text: str) -> dict[tuple[str, str, int, int], int]:
    reader = csv.DictReader(io.StringIO(text))
    required = {"T1", "T2", "FTG_T1", "FTG_T2"}
    if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
        raise ValueError("jleague2020_snapshot_missing_required_columns")

    counts: dict[tuple[str, str, int, int], int] = {}
    for raw in reader:
        try:
            home = str(raw.get("T1", "")).strip()
            away = str(raw.get("T2", "")).strip()
            hg = int(float(raw.get("FTG_T1", "")))
            ag = int(float(raw.get("FTG_T2", "")))
        except (TypeError, ValueError):
            continue
        if not home or not away or hg < 0 or ag < 0:
            continue
        key = (_norm(home), _norm(away), hg, ag)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _target_key(row: pd.Series) -> tuple[str, str, int, int] | None:
    try:
        home = str(row.get("home_team", "")).strip()
        away = str(row.get("away_team", "")).strip()
        hg = int(float(row.get("home_goals")))
        ag = int(float(row.get("away_goals")))
    except (TypeError, ValueError):
        return None
    if not home or not away or hg < 0 or ag < 0:
        return None
    return (_norm(home), _norm(away), hg, ag)


def apply_jleague_2020_github_pit(
    history: pd.DataFrame,
    *,
    cache_dir: str = "data/raw/pit_evidence",
    timeout: int = 30,
) -> pd.DataFrame:
    if history is None or history.empty:
        return history.copy() if history is not None else history

    mask = (
        history["competition"].astype(str).eq(COMPETITION)
        & pd.to_numeric(history.get("season_start"), errors="coerce").eq(SEASON_START)
    )
    if not mask.any():
        return history.iloc[0:0].copy()

    pending: list[tuple[Any, tuple[str, str, int, int], pd.Timestamp]] = []
    for idx, row in history.loc[mask].iterrows():
        key = _target_key(row)
        if key is None:
            continue
        if str(row.get("pit_evidence_status", "")) == "VERIFIED":
            continue
        lower_bound, reason = _result_lower_bound(row)
        if lower_bound is None:
            continue
        # DATE_ONLY rows use the next local-day boundary as the lower bound via
        # _result_lower_bound. This deliberately sacrifices same-day evidence.
        pending.append((idx, key, lower_bound))

    if not pending:
        return history.iloc[0:0].copy()

    commits = _commits(cache_dir, timeout=timeout)
    ordered = sorted(
        (
            (dt, str(commit["sha"]).strip())
            for commit in commits
            if (dt := _commit_time(commit)) is not None
        ),
        key=lambda x: x[0],
    )
    if not ordered:
        raise RuntimeError("jleague2020_commit_history_has_no_dated_commits")

    verified: dict[Any, dict[str, Any]] = {}
    remaining = list(pending)
    for commit_time, sha in ordered:
        eligible = [item for item in remaining if commit_time >= item[2]]
        if not eligible:
            continue
        text = _snapshot_text(sha, cache_dir, timeout=timeout)
        try:
            counts = _source_rows(text)
        except ValueError as exc:
            raise RuntimeError(f"jleague2020_snapshot_parse_failed:{exc}") from exc
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        evidence_url = f"https://github.com/{REPOSITORY}/blob/{sha}/Match%20Data/JLeague-2020.csv"

        newly_verified = set()
        for idx, key, _ in eligible:
            count = counts.get(key, 0)
            if count != 1:
                continue
            verified[idx] = {
                "source_available_at_utc": commit_time.isoformat(),
                "pit_evidence_status": "VERIFIED",
                "pit_evidence_reason": "immutable_jleague_2020_snapshot_contains_unique_exact_result",
                "pit_evidence_url": evidence_url,
                "capture_digest": digest,
            }
            newly_verified.add(idx)
        if newly_verified:
            remaining = [item for item in remaining if item[0] not in newly_verified]
        if not remaining:
            break

    if not verified:
        return history.iloc[0:0].copy()

    out = pd.DataFrame.from_dict(verified, orient="index")
    out.index.name = history.index.name
    return out
