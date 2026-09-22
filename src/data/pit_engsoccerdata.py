from __future__ import annotations

"""Conservative PIT evidence from immutable public Git snapshots.

A row is verified only when the exact completed-result identity is present in a
fixed immutable snapshot and the snapshot commit time is at/after the existing
conservative result-publication lower bound. This is evidence recovery only.
"""

import base64
import io
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from src.data.pit_source_adapter_fast import normalize_team_identity
from src.data.pit_source_adapter_v2 import _result_lower_bound

REPOSITORY = "jalapic/engsoccerdata"

SNAPSHOTS = {
    "EPL": {
        "path": "data-raw/england.csv",
        "commit_sha": "f34131cf85311c2fe0e681ab3811eb94acee330b",
        "blob_sha": "2472650a74a31a4a64e4a554a59e82c388f05151",
        "observed_at_utc": "2022-11-05T19:16:32+00:00",
    },
    "LL": {
        "path": "data-raw/spain.csv",
        "commit_sha": "f409c8bdfb7417883fd157b8398443b5cadc3d55",
        "blob_sha": "8d90011a646f1d16705eb452dedf3381af4bd298",
        "observed_at_utc": "2022-11-03T21:56:57+00:00",
    },
    "BL1": {
        "path": "data-raw/germany.csv",
        "commit_sha": "04bcec3219da6a944b17799bb0d66a85f4953e17",
        "blob_sha": "a426758401ec282c5bf24b50037e460647bc2db9",
        "observed_at_utc": "2022-11-04T18:58:20+00:00",
    },
    "SA": {
        "path": "data-raw/italy.csv",
        "commit_sha": "ab345a1c6a6df821872785c64d93e910bdac496b",
        "blob_sha": "37e5b702b25ef5aa9684cd97f181ada8841f67dd",
        "observed_at_utc": "2022-11-04T04:42:23+00:00",
    },
    "FL1": {
        "path": "data-raw/france.csv",
        "commit_sha": "880b4a9e7e89baa15d66252a19df2c636f2152f9",
        "blob_sha": "30805787f1ab1de79b5bc1225abf56e53a71125b",
        "observed_at_utc": "2022-11-04T15:11:07+00:00",
    },
    "ERE": {
        "path": "data-raw/holland.csv",
        "commit_sha": "cf06c5c6f918e558bd579feaa990c0cffe2ad42e",
        "blob_sha": "a84a96a234e7e90dadd3c802b2a277d1e0df33dc",
        "observed_at_utc": "2022-11-03T22:10:20+00:00",
    },
}

def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "SoccerPredictionResearch/PIT-Engsoccerdata",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

def _blob_text(blob_sha: str, timeout: int = 45) -> str:
    url = f"https://api.github.com/repos/{REPOSITORY}/git/blobs/{quote(blob_sha, safe='')}"
    response = requests.get(url, headers=_headers(), timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if payload.get("encoding") != "base64":
        raise ValueError("immutable snapshot blob was not returned as base64")
    return base64.b64decode(
        str(payload.get("content", "")).encode("ascii")
    ).decode("utf-8", errors="replace")

def _identity_key(date, home_team, away_team, home_goals, away_goals) -> tuple | None:
    try:
        dt = pd.Timestamp(date)
        if pd.isna(dt):
            return None
        hg = int(float(home_goals))
        ag = int(float(away_goals))
    except (TypeError, ValueError):
        return None
    result = "H" if hg > ag else "D" if hg == ag else "A"
    return (
        dt.date().isoformat(),
        normalize_team_identity(home_team),
        normalize_team_identity(away_team),
        float(hg),
        float(ag),
        result,
    )

def _snapshot_keys(text: str, competition: str) -> set[tuple]:
    frame = pd.read_csv(io.StringIO(text), low_memory=False)
    required = {"Date", "Season", "home", "visitor", "hgoal", "vgoal"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"{competition} snapshot missing columns: {sorted(required - set(frame.columns))}"
        )
    if "tier" in frame.columns:
        tier = pd.to_numeric(frame["tier"], errors="coerce")
        frame = frame.loc[tier.eq(1)].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Season"] = pd.to_numeric(frame["Season"], errors="coerce")
    frame = frame.loc[
        frame["Date"].notna()
        & frame["Season"].between(2010, 2025, inclusive="both")
    ]
    keys = set()
    for row in frame.itertuples(index=False):
        key = _identity_key(
            getattr(row, "Date", None),
            getattr(row, "home", None),
            getattr(row, "visitor", None),
            getattr(row, "hgoal", None),
            getattr(row, "vgoal", None),
        )
        if key is not None:
            keys.add(key)
    return keys

def apply_snapshot_pit(history: pd.DataFrame, *, cache_dir: str = "data/raw/pit_evidence", timeout: int = 45) -> pd.DataFrame:
    """Enrich only unverified rows with immutable snapshot evidence."""
    if history is None or history.empty:
        return history.copy() if history is not None else history
    out = history.copy()
    if "source_available_at_utc" not in out.columns:
        out["source_available_at_utc"] = pd.NaT
    for col in ("pit_evidence_status", "pit_evidence_reason", "pit_evidence_url", "capture_digest"):
        if col not in out.columns:
            out[col] = pd.NA

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    for competition, snapshot in SNAPSHOTS.items():
        mask = out["competition"].astype(str).eq(competition)
        if not mask.any():
            continue
        observed_at = datetime.fromisoformat(snapshot["observed_at_utc"]).astimezone(timezone.utc)
        cache_path = cache_root / f"engsoccerdata-{snapshot['blob_sha']}.csv"
        try:
            if cache_path.exists():
                raw_text = cache_path.read_text(encoding="utf-8")
            else:
                raw_text = _blob_text(snapshot["blob_sha"], timeout=timeout)
                cache_path.write_text(raw_text, encoding="utf-8")
            keys = _snapshot_keys(raw_text, competition)
        except Exception as exc:
            for idx in out.index[mask]:
                if str(out.at[idx, "pit_evidence_status"]) != "VERIFIED":
                    out.at[idx, "pit_evidence_status"] = "UNVERIFIABLE"
                    out.at[idx, "pit_evidence_reason"] = (
                        f"engsoccerdata_snapshot_error:{type(exc).__name__}:{exc}"
                    )
            continue

        for idx, row in out.loc[mask].iterrows():
            if str(out.at[idx, "pit_evidence_status"]) == "VERIFIED":
                continue
            key = _identity_key(
                row.get("source_event_date", row.get("kickoff_utc")),
                row.get("home_team"),
                row.get("away_team"),
                row.get("home_goals"),
                row.get("away_goals"),
            )
            lower_bound, bound_reason = _result_lower_bound(row)
            if key is None or lower_bound is None or key not in keys:
                continue
            lower_bound_dt = pd.Timestamp(lower_bound).to_pydatetime()
            if lower_bound_dt.tzinfo is None:
                lower_bound_dt = lower_bound_dt.replace(tzinfo=timezone.utc)
            else:
                lower_bound_dt = lower_bound_dt.astimezone(timezone.utc)
            if observed_at < lower_bound_dt:
                continue
            out.at[idx, "source_available_at_utc"] = observed_at.isoformat()
            out.at[idx, "pit_evidence_status"] = "VERIFIED"
            out.at[idx, "pit_evidence_reason"] = (
                "immutable_engsoccerdata_snapshot_exact_result_"
                f"after_{bound_reason.lower()}"
            )
            out.at[idx, "pit_evidence_url"] = (
                f"https://github.com/{REPOSITORY}/blob/"
                f"{snapshot['commit_sha']}/{snapshot['path']}"
            )
            out.at[idx, "capture_digest"] = snapshot["commit_sha"]
    return out
