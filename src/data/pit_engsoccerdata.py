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
TEAMNAMES_PATH = "data-raw/teamnames.csv"

SNAPSHOTS = [
    {
        "competition": "EPL",
        "path": "data-raw/england.csv",
        "commit_sha": "69b6f0eee32fcd9fc6c41d6bcf9959851d56100a",
        "blob_sha": "679d8e8093c98a156fcf2ec762a16df4a82065d0",
        "teamnames_blob_sha": "4528990fcdf5a39248961c474629ade444b40604",
        "country": "England",
        "observed_at_utc": "2017-01-14T04:13:26+00:00",
    },
    {
        "competition": "EPL",
        "path": "data-raw/england.csv",
        "commit_sha": "86c2a8df92e8f3bf76680b0dbfc97834d20f1f87",
        "blob_sha": "6377f7d0bd5aaf34c484be2cdb72d11a1483f77e",
        "teamnames_blob_sha": "572cceeb121a574768db0dda9833c40fd18ea794",
        "country": "England",
        "observed_at_utc": "2022-01-18T20:51:44+00:00",
    },
    {
        "competition": "EPL",
        "path": "data-raw/england.csv",
        "commit_sha": "f34131cf85311c2fe0e681ab3811eb94acee330b",
        "blob_sha": "2472650a74a31a4a64e4a554a59e82c388f05151",
        "teamnames_blob_sha": "572cceeb121a574768db0dda9833c40fd18ea794",
        "country": "England",
        "observed_at_utc": "2022-11-05T19:16:32+00:00",
    },
    {
        "competition": "LL",
        "path": "data-raw/spain.csv",
        "commit_sha": "d771707184c9b1ab65e1a9a0595b6da3aad9a505",
        "blob_sha": "66f23e5dfddf1223e5512dab380b49800f4d16db",
        "teamnames_blob_sha": "424f64e8fe941a25e9b96372d4a31e23e7615d7a",
        "country": "Spain",
        "observed_at_utc": "2020-10-24T18:48:54+00:00",
    },
    {
        "competition": "LL",
        "path": "data-raw/spain.csv",
        "commit_sha": "f409c8bdfb7417883fd157b8398443b5cadc3d55",
        "blob_sha": "8d90011a646f1d16705eb452dedf3381af4bd298",
        "teamnames_blob_sha": "9333cdc38a07e7f03c854859b5a0d75edf329955",
        "country": "Spain",
        "observed_at_utc": "2022-11-03T21:56:57+00:00",
    },
    {
        "competition": "BL1",
        "path": "data-raw/germany.csv",
        "commit_sha": "a899960664a127af2aa3f2858b2b28cc0f0e4317",
        "blob_sha": "4bfd9101a05c72b57b77b44d1b3e7801e4509214",
        "teamnames_blob_sha": "956f215c7deab5125864df843b66e1caca7764d6",
        "country": "Germany",
        "observed_at_utc": "2017-05-15T16:30:20+00:00",
    },
    {
        "competition": "BL1",
        "path": "data-raw/germany.csv",
        "commit_sha": "257a3ec253badbce350dc7a72710574200ca0d42",
        "blob_sha": "67befb4f1194b2a93f21b8325e50e2a82ec80516",
        "teamnames_blob_sha": "eae2d295f89bcebd01cf16ee6cab4d7dc3783e02",
        "country": "Germany",
        "observed_at_utc": "2020-10-23T03:22:16+00:00",
    },
    {
        "competition": "BL1",
        "path": "data-raw/germany.csv",
        "commit_sha": "04bcec3219da6a944b17799bb0d66a85f4953e17",
        "blob_sha": "a426758401ec282c5bf24b50037e460647bc2db9",
        "teamnames_blob_sha": "a55fbc1b98ad9570ba5f6c6295b72275357e0381",
        "country": "Germany",
        "observed_at_utc": "2022-11-04T18:58:20+00:00",
    },
    {
        "competition": "SA",
        "path": "data-raw/italy.csv",
        "commit_sha": "cdc7008bec7781baea13aa3cee544c4a7e263c12",
        "blob_sha": "c1f03db8d2a4f2e2440243c3d6079ac87f5c714e",
        "teamnames_blob_sha": "a4099eb6a4940231da6f75ccb87ab29f4bbd547b",
        "country": "Italy",
        "observed_at_utc": "2020-10-24T04:26:18+00:00",
    },
    {
        "competition": "SA",
        "path": "data-raw/italy.csv",
        "commit_sha": "ab345a1c6a6df821872785c64d93e910bdac496b",
        "blob_sha": "37e5b702b25ef5aa9684cd97f181ada8841f67dd",
        "teamnames_blob_sha": "78a1a9c663b738d1af20e6a99d9a31aab684abdb",
        "country": "Italy",
        "observed_at_utc": "2022-11-04T04:42:23+00:00",
    },
    {
        "competition": "FL1",
        "path": "data-raw/france.csv",
        "commit_sha": "9012bf9f0f2fd59c1d0b977f90be7b9d92c5cfd4",
        "blob_sha": "6701c14212e57795a8b0890450e750a5d6f142d4",
        "teamnames_blob_sha": "4528990fcdf5a39248961c474629ade444b40604",
        "country": "France",
        "observed_at_utc": "2017-01-14T03:52:17+00:00",
    },
    {
        "competition": "FL1",
        "path": "data-raw/france.csv",
        "commit_sha": "49af5838849beb2068578fedf66228c9e2177fd2",
        "blob_sha": "67befb4f1194b2a93f21b8325e50e2a82ec80516",
        "teamnames_blob_sha": "804a58cfd6968b705d3850d08594827e863a2acb",
        "country": "France",
        "observed_at_utc": "2020-10-23T04:25:45+00:00",
    },
    {
        "competition": "FL1",
        "path": "data-raw/france.csv",
        "commit_sha": "880b4a9e7e89baa15d66252a19df2c636f2152f9",
        "blob_sha": "30805787f1ab1de79b5bc1225abf56e53a71125b",
        "teamnames_blob_sha": "a55fbc1b98ad9570ba5f6c6295b72275357e0381",
        "country": "France",
        "observed_at_utc": "2022-11-04T15:11:07+00:00",
    },
    {
        "competition": "ERE",
        "path": "data-raw/holland.csv",
        "commit_sha": "bda660d02aa3ed1e1fcbef01fbdbd5b1c816898d",
        "blob_sha": "6d34538e20cc8e51968eddd45ba0aa615b832f6a",
        "teamnames_blob_sha": "d8191be6c49f169a6b249e681e85c8e00dbdae3a",
        "country": None,
        "observed_at_utc": "2020-10-24T06:11:43+00:00",
    },
    {
        "competition": "ERE",
        "path": "data-raw/holland.csv",
        "commit_sha": "cf06c5c6f918e558bd579feaa990c0cffe2ad42e",
        "blob_sha": "a84a96a234e7e90dadd3c802b2a277d1e0df33dc",
        "teamnames_blob_sha": "73e61675b3d24989f03c906242e58c99583e7aee",
        "country": None,
        "observed_at_utc": "2022-11-03T22:10:20+00:00",
    },
]



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


def _identity_key(
    date,
    home_team,
    away_team,
    home_goals,
    away_goals,
    *,
    aliases: dict[str, str] | None = None,
) -> tuple | None:
    try:
        dt = pd.Timestamp(date)
        if pd.isna(dt):
            return None
        hg = int(float(home_goals))
        ag = int(float(away_goals))
    except (TypeError, ValueError):
        return None

    def canonical(value: object) -> str:
        normalized = normalize_team_identity(value)
        return aliases.get(normalized, normalized) if aliases is not None else normalized

    result = "H" if hg > ag else "D" if hg == ag else "A"
    return (
        dt.date().isoformat(),
        canonical(home_team),
        canonical(away_team),
        float(hg),
        float(ag),
        result,
    )


def _team_aliases(teamnames_text: str, country: str | None = None) -> dict[str, str]:
    """Build only unambiguous immutable team-name aliases from the same snapshot."""
    frame = pd.read_csv(io.StringIO(teamnames_text), low_memory=False)
    required = {"country", "name", "name_other"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"teamnames snapshot missing columns: {sorted(required - set(frame.columns))}"
        )
    if country:
        frame = frame.loc[frame["country"].astype(str).eq(country)].copy()

    candidates: dict[str, set[str]] = {}
    for row in frame.itertuples(index=False):
        canonical = normalize_team_identity(getattr(row, "name", ""))
        if not canonical:
            continue
        for raw in (getattr(row, "name", ""), getattr(row, "name_other", "")):
            if raw is None or str(raw).strip().lower() in {"", "nan", "na", "<na>"}:
                continue
            alias = normalize_team_identity(raw)
            if not alias:
                continue
            candidates.setdefault(alias, set()).add(canonical)

    # Ambiguous aliases stay unresolved instead of guessing across clubs.
    return {
        alias: next(iter(values))
        for alias, values in candidates.items()
        if len(values) == 1
    }


def _snapshot_keys(
    text: str,
    competition: str,
    *,
    aliases: dict[str, str] | None = None,
) -> set[tuple]:
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
            aliases=aliases,
        )
        if key is not None:
            keys.add(key)
    return keys


def apply_snapshot_pit(
    history: pd.DataFrame,
    *,
    cache_dir: str = "data/raw/pit_evidence",
    timeout: int = 45,
) -> pd.DataFrame:
    """Enrich only unverified rows with immutable snapshot evidence."""
    if history is None or history.empty:
        return history.copy() if history is not None else history
    out = history.copy()
    if "source_available_at_utc" not in out.columns:
        out["source_available_at_utc"] = pd.NaT
    for col in (
        "pit_evidence_status",
        "pit_evidence_reason",
        "pit_evidence_url",
        "capture_digest",
    ):
        if col not in out.columns:
            out[col] = pd.NA

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    alias_cache: dict[str, dict[str, str]] = {}

    for snapshot in sorted(SNAPSHOTS, key=lambda item: item["observed_at_utc"]):
        competition = str(snapshot["competition"])
        mask = out["competition"].astype(str).eq(competition)
        if not mask.any():
            continue

        observed_at = datetime.fromisoformat(
            snapshot["observed_at_utc"]
        ).astimezone(timezone.utc)
        cache_path = cache_root / f"engsoccerdata-{snapshot['blob_sha']}.csv"
        teamnames_cache_path = (
            cache_root / f"engsoccerdata-teamnames-{snapshot['teamnames_blob_sha']}.csv"
        )
        try:
            if cache_path.exists():
                raw_text = cache_path.read_text(encoding="utf-8")
            else:
                raw_text = _blob_text(snapshot["blob_sha"], timeout=timeout)
                cache_path.write_text(raw_text, encoding="utf-8")

            alias_key = snapshot["teamnames_blob_sha"]
            if alias_key not in alias_cache:
                if teamnames_cache_path.exists():
                    teamnames_text = teamnames_cache_path.read_text(encoding="utf-8")
                else:
                    teamnames_text = _blob_text(alias_key, timeout=timeout)
                    teamnames_cache_path.write_text(teamnames_text, encoding="utf-8")
                alias_cache[alias_key] = _team_aliases(
                    teamnames_text,
                    country=(
                        str(snapshot.get("country")).strip()
                        if snapshot.get("country") is not None
                        and str(snapshot.get("country")).strip()
                        else None
                    ),
                )

            aliases = alias_cache[alias_key]
            keys = _snapshot_keys(raw_text, competition, aliases=aliases)
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
                aliases=aliases,
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
                f"after_{bound_reason.lower()}_with_snapshot_team_alias_reconciliation"
            )
            out.at[idx, "pit_evidence_url"] = (
                f"https://github.com/{REPOSITORY}/blob/"
                f"{snapshot['commit_sha']}/{snapshot['path']}"
            )
            out.at[idx, "capture_digest"] = snapshot["commit_sha"]

    return out
