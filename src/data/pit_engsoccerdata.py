from __future__ import annotations

"""Conservative PIT evidence from immutable public Git snapshots.

A row is verified only when the exact completed-result identity is present in an
immutable snapshot whose commit time is at/after the existing conservative
result-publication lower bound. Multiple historical snapshots are used so the
earliest valid observation can be recovered without treating a later snapshot
as if it were available earlier.
"""

import io
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.data.pit_source_adapter_fast import normalize_team_identity
from src.data.pit_source_adapter_v2 import _result_lower_bound

REPOSITORY = "jalapic/engsoccerdata"

# Immutable source snapshots discovered from the repository history.  More than
# one snapshot per competition is intentional: it increases PIT-recoverable
# coverage while preserving the exact observed publication-proxy timestamp.
SNAPSHOTS = {
    "EPL": [
        {"path": "data-raw/england.csv", "commit_sha": "df47452e020bcf996ae7b92385fa0dbcf76767a2", "observed_at_utc": "2016-05-16T03:12:20+00:00"},
        {"path": "data-raw/england.csv", "commit_sha": "55a7bf08dad01c6bd330a0249ae1819a26d09c99", "observed_at_utc": "2017-10-01T22:16:27+00:00"},
        {"path": "data-raw/england.csv", "commit_sha": "aefaea3d194afb66e19a2695d15d5e5a24cfd1f5", "observed_at_utc": "2020-10-17T21:07:43+00:00"},
        {"path": "data-raw/england.csv", "commit_sha": "f34131cf85311c2fe0e681ab3811eb94acee330b", "observed_at_utc": "2022-11-05T19:16:32+00:00"},
    ],
    "LL": [
        {"path": "data-raw/spain.csv", "commit_sha": "df47452e020bcf996ae7b92385fa0dbcf76767a2", "observed_at_utc": "2016-05-16T03:12:20+00:00"},
        {"path": "data-raw/spain.csv", "commit_sha": "55a7bf08dad01c6bd330a0249ae1819a26d09c99", "observed_at_utc": "2017-10-01T22:16:27+00:00"},
        {"path": "data-raw/spain.csv", "commit_sha": "d771707184c9b1ab65e1a9a0595b6da3aad9a505", "observed_at_utc": "2020-10-24T18:48:54+00:00"},
        {"path": "data-raw/spain.csv", "commit_sha": "f409c8bdfb7417883fd157b8398443b5cadc3d55", "observed_at_utc": "2022-11-03T21:56:57+00:00"},
    ],
    "BL1": [
        {"path": "data-raw/germany.csv", "commit_sha": "df47452e020bcf996ae7b92385fa0dbcf76767a2", "observed_at_utc": "2016-05-16T03:12:20+00:00"},
        {"path": "data-raw/germany.csv", "commit_sha": "55a7bf08dad01c6bd330a0249ae1819a26d09c99", "observed_at_utc": "2017-10-01T22:16:27+00:00"},
        {"path": "data-raw/germany.csv", "commit_sha": "257a3ec253badbce350dc7a72710574200ca0d42", "observed_at_utc": "2020-10-23T03:22:16+00:00"},
        {"path": "data-raw/germany.csv", "commit_sha": "04bcec3219da6a944b17799bb0d66a85f4953e17", "observed_at_utc": "2022-11-04T18:58:20+00:00"},
    ],
    "SA": [
        {"path": "data-raw/italy.csv", "commit_sha": "df47452e020bcf996ae7b92385fa0dbcf76767a2", "observed_at_utc": "2016-05-16T03:12:20+00:00"},
        {"path": "data-raw/italy.csv", "commit_sha": "55a7bf08dad01c6bd330a0249ae1819a26d09c99", "observed_at_utc": "2017-10-01T22:16:27+00:00"},
        {"path": "data-raw/italy.csv", "commit_sha": "57a5d0cfdf247db480ed9c68ab5b7cff8c263c5e", "observed_at_utc": "2018-08-27T18:37:59+00:00"},
        {"path": "data-raw/italy.csv", "commit_sha": "f79c690b69f5cdd4217d3df7597b543823eed1ee", "observed_at_utc": "2019-08-23T14:28:17+00:00"},
        {"path": "data-raw/italy.csv", "commit_sha": "58edbe38504eaefefac0f5fcfe2ee38841106bba", "observed_at_utc": "2020-10-24T16:57:03+00:00"},
        {"path": "data-raw/italy.csv", "commit_sha": "ab345a1c6a6df821872785c64d93e910bdac496b", "observed_at_utc": "2022-11-04T04:42:23+00:00"},
    ],
    "FL1": [
        {"path": "data-raw/france.csv", "commit_sha": "df47452e020bcf996ae7b92385fa0dbcf76767a2", "observed_at_utc": "2016-05-16T03:12:20+00:00"},
        {"path": "data-raw/france.csv", "commit_sha": "55a7bf08dad01c6bd330a0249ae1819a26d09c99", "observed_at_utc": "2017-10-01T22:16:27+00:00"},
        {"path": "data-raw/france.csv", "commit_sha": "49af5838849beb2068578fedf66228c9e2177fd2", "observed_at_utc": "2020-10-23T04:25:45+00:00"},
        {"path": "data-raw/france.csv", "commit_sha": "880b4a9e7e89baa15d66252a19df2c636f2152f9", "observed_at_utc": "2022-11-04T15:11:07+00:00"},
    ],
    "ERE": [
        {"path": "data-raw/holland.csv", "commit_sha": "df47452e020bcf996ae7b92385fa0dbcf76767a2", "observed_at_utc": "2016-05-16T03:12:20+00:00"},
        {"path": "data-raw/holland.csv", "commit_sha": "55a7bf08dad01c6bd330a0249ae1819a26d09c99", "observed_at_utc": "2017-10-01T22:16:27+00:00"},
        {"path": "data-raw/holland.csv", "commit_sha": "bda660d02aa3ed1e1fcbef01fbdbd5b1c816898d", "observed_at_utc": "2020-10-24T04:11:43+00:00"},
        {"path": "data-raw/holland.csv", "commit_sha": "cf06c5c6f918e558bd579feaa990c0cffe2ad42e", "observed_at_utc": "2022-11-03T22:10:20+00:00"},
    ],
}

def _utc(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _snapshot_text(snapshot: dict[str, str], cache_dir: str, timeout: int) -> str:
    commit_sha = str(snapshot["commit_sha"])
    path = str(snapshot["path"])
    cache_path = Path(cache_dir) / f"engsoccerdata-{commit_sha}.csv"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    url = f"https://raw.githubusercontent.com/{REPOSITORY}/{commit_sha}/{path}"
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "SoccerPredictionResearch/PIT-Engsoccerdata"},
    )
    response.raise_for_status()
    raw = response.content
    if raw.lstrip().lower().startswith((b"<!doctype html", b"<html")):
        raise ValueError("immutable snapshot returned HTML instead of CSV")
    text = raw.decode("utf-8", errors="replace")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    return text

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

def _select_snapshot(key: tuple, lower_bound: datetime, snapshots_with_keys: list[tuple[datetime, dict[str, str], set[tuple]]]) -> tuple[datetime, dict[str, str]] | None:
    for observed_at, snapshot, keys in sorted(snapshots_with_keys, key=lambda x: x[0]):
        if observed_at >= lower_bound and key in keys:
            return observed_at, snapshot
    return None

def apply_snapshot_pit(
    history: pd.DataFrame,
    *,
    cache_dir: str = "data/raw/pit_evidence",
    timeout: int = 45,
) -> pd.DataFrame:
    """Enrich only unverified rows using the earliest valid immutable snapshot."""
    if history is None or history.empty:
        return history.copy() if history is not None else history
    out = history.copy()
    if "source_available_at_utc" not in out.columns:
        out["source_available_at_utc"] = pd.NaT
    for col in ("pit_evidence_status", "pit_evidence_reason", "pit_evidence_url", "capture_digest"):
        if col not in out.columns:
            out[col] = pd.NA

    for competition, snapshots in SNAPSHOTS.items():
        mask = out["competition"].astype(str).eq(competition)
        if not mask.any():
            continue
        parsed: list[tuple[datetime, dict[str, str], set[tuple]]] = []
        for snapshot in sorted(snapshots, key=lambda x: x["observed_at_utc"]):
            observed_at = _utc(snapshot["observed_at_utc"])
            if observed_at is None:
                continue
            try:
                keys = _snapshot_keys(_snapshot_text(snapshot, cache_dir, timeout), competition)
            except Exception as exc:
                # One bad historical snapshot must not invalidate independent
                # snapshots; unresolved rows remain fail-closed.
                continue
            parsed.append((observed_at, snapshot, keys))
        if not parsed:
            for idx in out.index[mask]:
                if str(out.at[idx, "pit_evidence_status"]) != "VERIFIED":
                    out.at[idx, "pit_evidence_status"] = "UNVERIFIABLE"
                    out.at[idx, "pit_evidence_reason"] = "no_usable_immutable_snapshot"
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
            lower_bound_dt = _utc(lower_bound)
            if key is None or lower_bound_dt is None:
                continue
            selected = _select_snapshot(key, lower_bound_dt, parsed)
            if selected is None:
                continue
            observed_at, snapshot = selected
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
