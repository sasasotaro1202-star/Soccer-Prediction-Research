from __future__ import annotations

"""Point-in-time source evidence adapters for the soccer research system.

The key PIT distinction is that a historical match result is not expected to be
available before that match. We therefore determine when each *past result*
first appears in an archived source snapshot, then let the feature builder use
that result only for predictions whose cutoff is at or after that availability
time. A current retrieval timestamp is never promoted to source availability.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.data.football_data import BASE, season_folder

WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"
WAYBACK_WEB = "https://web.archive.org/web"

COMPETITION_ADAPTERS = {
    "E0": {"source": "Football-Data.co.uk", "source_code": "E0", "adapter": "football_data_wayback"},
    "CH": {"source": "Football-Data.co.uk", "source_code": "E1", "adapter": "football_data_wayback"},
    "D1": {"source": "Football-Data.co.uk", "source_code": "D1", "adapter": "football_data_wayback"},
    "I1": {"source": "Football-Data.co.uk", "source_code": "I1", "adapter": "football_data_wayback"},
    "SP1": {"source": "Football-Data.co.uk", "source_code": "SP1", "adapter": "football_data_wayback"},
    "F1": {"source": "Football-Data.co.uk", "source_code": "F1", "adapter": "football_data_wayback"},
    "N1": {"source": "Football-Data.co.uk", "source_code": "N1", "adapter": "football_data_wayback"},
    "UCL": {"source": "UNVERIFIED", "source_code": None, "adapter": None},
    "UEL": {"source": "UNVERIFIED", "source_code": None, "adapter": None},
    "J1": {"source": "UNVERIFIED", "source_code": None, "adapter": None},
    "J2": {"source": "UNVERIFIED", "source_code": None, "adapter": None},
    "J3": {"source": "UNVERIFIED", "source_code": None, "adapter": None},
    "DFBP": {"source": "UNVERIFIED", "source_code": None, "adapter": None},
    "FRIENDLY": {"source": "UNVERIFIED", "source_code": None, "adapter": None},
    "EFL": {"source": "UNVERIFIED", "source_code": None, "adapter": None},
}


@dataclass(frozen=True)
class SourceEvidence:
    source_available_at_utc: str | None
    evidence_status: str
    evidence_url: str | None = None
    capture_digest: str | None = None
    reason: str = ""


def _utc(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def source_url(competition: str, start_year: int) -> str:
    spec = COMPETITION_ADAPTERS.get(competition)
    if not spec or not spec["source_code"]:
        raise ValueError(f"No PIT source mapping for {competition}")
    return BASE.format(season_folder=season_folder(start_year), league=spec["source_code"])


class FootballDataWaybackAdapter:
    """Find the first archived snapshot containing a completed match result."""

    def __init__(self, cache_dir: str = "data/raw/pit_evidence", timeout: int = 30):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "SoccerPredictionResearch/1.0 PIT-Audit"})
        self._captures: dict[str, list[dict[str, str]]] = {}
        self._snapshot_rows: dict[str, pd.DataFrame] = {}
        self._snapshot_keys: dict[str, set[tuple]] = {}

    @staticmethod
    def _cache_key(url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()

    def captures(self, url: str) -> list[dict[str, str]]:
        if url in self._captures:
            return self._captures[url]
        cache = self.cache_dir / f"captures_{self._cache_key(url)}.json"
        if cache.exists():
            rows = json.loads(cache.read_text(encoding="utf-8"))
            self._captures[url] = rows
            return rows
        params = {
            "url": url,
            "output": "json",
            "filter": "statuscode:200",
            "fl": "timestamp,digest,original,statuscode,mimetype",
            "collapse": "digest",
        }
        try:
            response = self.session.get(WAYBACK_CDX, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            self._captures[url] = []
            return []
        if not payload or len(payload) <= 1:
            rows = []
        else:
            headers, *data = payload
            rows = [dict(zip(headers, row)) for row in data]
        cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        self._captures[url] = rows
        return rows

    @staticmethod
    def _snapshot_url(capture: dict[str, str], original_url: str) -> str:
        return f"{WAYBACK_WEB}/{capture['timestamp']}id_/{original_url}"

    def _snapshot_frame(self, capture: dict[str, str], original_url: str) -> pd.DataFrame | None:
        digest = capture.get("digest") or capture.get("timestamp", "")
        if digest in self._snapshot_rows:
            return self._snapshot_rows[digest]
        try:
            response = self.session.get(self._snapshot_url(capture, original_url), timeout=self.timeout)
            response.raise_for_status()
            frame = pd.read_csv(BytesIO(response.content))
        except Exception:
            return None
        self._snapshot_rows[digest] = frame
        return frame

    def _snapshot_keyset(self, capture: dict[str, str], original_url: str) -> set[tuple] | None:
        digest = capture.get("digest") or capture.get("timestamp", "")
        if digest in self._snapshot_keys:
            return self._snapshot_keys[digest]
        frame = self._snapshot_frame(capture, original_url)
        if frame is None:
            return None
        required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
        if not required.issubset(frame.columns):
            return None
        keys: set[tuple] = set()
        for r in frame.itertuples(index=False):
            try:
                date = pd.to_datetime(getattr(r, "Date"), dayfirst=True, errors="coerce")
                if pd.isna(date):
                    continue
                keys.add((
                    date.date().isoformat(),
                    str(getattr(r, "HomeTeam")).strip(),
                    str(getattr(r, "AwayTeam")).strip(),
                    float(getattr(r, "FTHG")),
                    float(getattr(r, "FTAG")),
                    str(getattr(r, "FTR")).strip(),
                ))
            except (TypeError, ValueError):
                continue
        self._snapshot_keys[digest] = keys
        return keys

    @staticmethod
    def _row_key(row: pd.Series) -> tuple | None:
        kickoff = _utc(row.get("kickoff_utc"))
        if kickoff is None:
            return None
        try:
            return (
                kickoff.date().isoformat(),
                str(row.get("home_team", "")).strip(),
                str(row.get("away_team", "")).strip(),
                float(row.get("home_goals")),
                float(row.get("away_goals")),
                str(row.get("result", "")).strip(),
            )
        except (TypeError, ValueError):
            return None

    def evidence_for_row(self, row: pd.Series) -> SourceEvidence:
        """Return the first source capture after the match containing its result."""
        competition = str(row.get("competition", ""))
        try:
            start_year = int(str(row.get("season", "0000/00")).split("/")[0])
        except ValueError:
            return SourceEvidence(None, "UNVERIFIABLE", reason="invalid_season")
        event = _utc(row.get("kickoff_utc"))
        key = self._row_key(row)
        if event is None or key is None:
            return SourceEvidence(None, "UNVERIFIABLE", reason="missing_record_identity")
        try:
            url = source_url(competition, start_year)
        except Exception as exc:
            return SourceEvidence(None, "UNVERIFIABLE", reason=str(exc))

        captures = []
        for capture in self.captures(url):
            ts = _utc(capture.get("timestamp"))
            if ts is not None and ts >= event:
                captures.append((ts, capture))
        captures.sort(key=lambda item: item[0])
        for ts, capture in captures:
            keys = self._snapshot_keyset(capture, url)
            if keys is not None and key in keys:
                return SourceEvidence(
                    source_available_at_utc=ts.isoformat(),
                    evidence_status="VERIFIED",
                    evidence_url=self._snapshot_url(capture, url),
                    capture_digest=capture.get("digest"),
                    reason="archived_completed_result_first_observed_after_event",
                )
        return SourceEvidence(None, "UNVERIFIABLE", reason="no_archive_snapshot_contains_completed_result")


def apply_pit_evidence(history: pd.DataFrame, *, cache_dir: str = "data/raw/pit_evidence") -> pd.DataFrame:
    """Annotate each historical result with independently evidenced availability."""
    if history.empty:
        return history.copy()
    out = history.copy()
    adapter = FootballDataWaybackAdapter(cache_dir=cache_dir)
    evidence = [adapter.evidence_for_row(row) for _, row in out.iterrows()]
    out["source_available_at_utc"] = [e.source_available_at_utc for e in evidence]
    out["pit_evidence_status"] = [e.evidence_status for e in evidence]
    out["pit_evidence_url"] = [e.evidence_url for e in evidence]
    out["pit_evidence_digest"] = [e.capture_digest for e in evidence]
    out["pit_evidence_reason"] = [e.reason for e in evidence]
    return out


def competition_adapter_matrix() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "competition": competition,
            "source": spec["source"],
            "adapter": spec["adapter"] or "NONE",
            "status": "IMPLEMENTED" if spec["adapter"] else "UNVERIFIED",
        }
        for competition, spec in COMPETITION_ADAPTERS.items()
    ])
