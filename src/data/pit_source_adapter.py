from __future__ import annotations

"""Point-in-time source evidence adapters for the soccer research system.

The adapter is deliberately fail-closed. A current retrieval timestamp is never
promoted to a historical availability timestamp. For Football-Data.co.uk
historical CSVs, Internet Archive captures are used as auditable evidence that
the source file was accessible no later than the capture time. Record-level
presence is also checked against the archived snapshot before a row is marked
PIT-verified.
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

from src.data.football_data import BASE, LEAGUES, season_folder

WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"
WAYBACK_WEB = "https://web.archive.org/web"

# The fixed 15-competition universe. Only sources with a concrete, auditable
# adapter are marked implemented; API existence alone never becomes coverage.
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
    """Resolve record-level PIT evidence from archived source snapshots."""

    def __init__(self, cache_dir: str = "data/raw/pit_evidence", timeout: int = 30):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "SoccerPredictionResearch/1.0 PIT-Audit"})
        self._captures: dict[str, list[dict[str, str]]] = {}
        self._snapshot_rows: dict[str, pd.DataFrame] = {}

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
        url = self._snapshot_url(capture, original_url)
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            frame = pd.read_csv(BytesIO(response.content))
        except Exception:
            return None
        self._snapshot_rows[digest] = frame
        return frame

    @staticmethod
    def _row_present(frame: pd.DataFrame, row: pd.Series) -> bool:
        required = {"HomeTeam", "AwayTeam", "Date"}
        if not required.issubset(frame.columns):
            return False
        home = str(row.get("home_team", "")).strip()
        away = str(row.get("away_team", "")).strip()
        kickoff = _utc(row.get("kickoff_utc"))
        if not home or not away or kickoff is None:
            return False
        dates = pd.to_datetime(frame["Date"], dayfirst=True, errors="coerce", utc=True)
        mask = frame["HomeTeam"].astype(str).str.strip().eq(home)
        mask &= frame["AwayTeam"].astype(str).str.strip().eq(away)
        mask &= dates.dt.date.eq(kickoff.date())
        return bool(mask.any())

    def evidence_for_row(self, row: pd.Series) -> SourceEvidence:
        competition = str(row.get("competition", ""))
        season_text = str(row.get("season", "0000/00"))
        try:
            start_year = int(season_text.split("/")[0])
        except ValueError:
            return SourceEvidence(None, "UNVERIFIABLE", reason="invalid_season")
        cutoff = _utc(row.get("prediction_cutoff_at_utc"))
        if cutoff is None:
            return SourceEvidence(None, "UNVERIFIABLE", reason="missing_prediction_cutoff")
        try:
            url = source_url(competition, start_year)
        except Exception as exc:
            return SourceEvidence(None, "UNVERIFIABLE", reason=str(exc))

        eligible = []
        for capture in self.captures(url):
            ts = _utc(capture.get("timestamp"))
            if ts is not None and ts <= cutoff:
                eligible.append((ts, capture))
        eligible.sort(key=lambda item: item[0], reverse=True)

        for ts, capture in eligible:
            frame = self._snapshot_frame(capture, url)
            if frame is not None and self._row_present(frame, row):
                return SourceEvidence(
                    source_available_at_utc=ts.isoformat(),
                    evidence_status="VERIFIED",
                    evidence_url=self._snapshot_url(capture, url),
                    capture_digest=capture.get("digest"),
                    reason="archived_source_snapshot_before_cutoff_contains_record",
                )
        if eligible:
            return SourceEvidence(None, "UNVERIFIABLE", reason="archive_capture_before_cutoff_without_record_match")
        return SourceEvidence(None, "UNVERIFIABLE", reason="no_archive_capture_before_cutoff")


def apply_pit_evidence(history: pd.DataFrame, *, cache_dir: str = "data/raw/pit_evidence") -> pd.DataFrame:
    """Add source-availability evidence without inventing timestamps."""
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
    rows = []
    for competition, spec in COMPETITION_ADAPTERS.items():
        rows.append({
            "competition": competition,
            "source": spec["source"],
            "adapter": spec["adapter"] or "NONE",
            "status": "IMPLEMENTED" if spec["adapter"] else "UNVERIFIED",
        })
    return pd.DataFrame(rows)
