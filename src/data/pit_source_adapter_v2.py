from __future__ import annotations

"""Leakage-safe Football-Data.co.uk PIT replay via Wayback CDX/archive."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.data.football_data import BASE, season_folder

WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"
WAYBACK_WEB = "https://web.archive.org/web"
DEFAULT_MAX_WORKERS = 8

COMPETITION_ADAPTERS = {
    "E0": {"source": "Football-Data.co.uk", "source_code": "E0", "adapter": "football_data_wayback"},
    "CH": {"source": "Football-Data.co.uk", "source_code": "E1", "adapter": "football_data_wayback"},
    "D1": {"source": "Football-Data.co.uk", "source_code": "D1", "adapter": "football_data_wayback"},
    "I1": {"source": "Football-Data.co.uk", "source_code": "I1", "adapter": "football_data_wayback"},
    "SP1": {"source": "Football-Data.co.uk", "source_code": "SP1", "adapter": "football_data_wayback"},
    "F1": {"source": "Football-Data.co.uk", "source_code": "F1", "adapter": "football_data_wayback"},
    "N1": {"source": "Football-Data.co.uk", "source_code": "N1", "adapter": "football_data_wayback"},
    "UCL": {"source": "UNVERIFIED", "source_code": None, "adapter": None}, "UEL": {"source": "UNVERIFIED", "source_code": None, "adapter": None},
    "J1": {"source": "UNVERIFIED", "source_code": None, "adapter": None}, "J2": {"source": "UNVERIFIED", "source_code": None, "adapter": None}, "J3": {"source": "UNVERIFIED", "source_code": None, "adapter": None},
    "DFBP": {"source": "UNVERIFIED", "source_code": None, "adapter": None}, "FRIENDLY": {"source": "UNVERIFIED", "source_code": None, "adapter": None}, "EFL": {"source": "UNVERIFIED", "source_code": None, "adapter": None},
}
INPUT_TO_FIXED = {"EPL": "E0", "CHA": "CH", "BL1": "D1", "SA": "I1", "LL": "SP1", "FL1": "F1", "ERE": "N1"}

@dataclass(frozen=True)
class SourceEvidence:
    source_available_at_utc: str | None
    evidence_status: str
    evidence_url: str | None = None
    capture_digest: str | None = None
    reason: str = ""

@dataclass(frozen=True)
class CaptureDiagnostic:
    status: str
    capture_count: int = 0
    error_type: str | None = None
    error: str | None = None

@dataclass(frozen=True)
class SnapshotDiagnostic:
    status: str
    keys: set[tuple] | None = None
    error_type: str | None = None
    error: str | None = None


def _utc(value: Any) -> datetime | None:
    if value is None or value == "" or pd.isna(value): return None
    text = str(value).strip()
    for parser in (lambda x: datetime.fromisoformat(x.replace("Z", "+00:00")), lambda x: datetime.strptime(x, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)):
        try:
            dt = parser(text)
            if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except (TypeError, ValueError):
            pass
    return None


def source_url(competition: str, start_year: int) -> str:
    fixed = INPUT_TO_FIXED.get(competition, competition); spec = COMPETITION_ADAPTERS.get(fixed)
    if not spec or not spec["source_code"]: raise ValueError(f"No PIT source mapping for {competition}")
    return BASE.format(season_folder=season_folder(start_year), league=spec["source_code"])


def _date_key(value: Any) -> str | None:
    if value is None or value == "" or pd.isna(value): return None
    text = str(value).strip()
    # ISO calendar dates must be parsed explicitly before day-first parsing.
    # This prevents YYYY-MM-DD from being silently reordered.
    for parser in (
        lambda x: pd.to_datetime(x, format="%Y-%m-%d", errors="coerce"),
        lambda x: pd.to_datetime(x, dayfirst=True, errors="coerce"),
    ):
        parsed = parser(text)
        if not pd.isna(parsed):
            return parsed.date().isoformat()
    return None


def _source_date_key(row: pd.Series) -> str | None:
    source_date = row.get("source_event_date")
    key = _date_key(source_date)
    if key is not None: return key
    return _date_key(row.get("kickoff_utc"))


def _row_key(row: pd.Series) -> tuple | None:
    date = _source_date_key(row)
    if date is None: return None
    try:
        return (date, str(row.get("home_team", "")).strip(), str(row.get("away_team", "")).strip(), float(row.get("home_goals")), float(row.get("away_goals")), str(row.get("result", "")).strip())
    except (TypeError, ValueError): return None


def _result_lower_bound(row: pd.Series) -> tuple[datetime | None, str]:
    kickoff = _utc(row.get("kickoff_utc"))
    if kickoff is None: return None, "MISSING_EVENT_TIME"
    if bool(row.get("kickoff_time_available", False)): return kickoff + timedelta(minutes=180), "KICKOFF_PLUS_180M"
    return kickoff.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1), "DATE_ONLY_NEXT_DAY"


class FootballDataWaybackAdapter:
    def __init__(self, cache_dir: str = "data/raw/pit_evidence", timeout: int = 30, max_workers: int = DEFAULT_MAX_WORKERS):
        self.cache_dir = Path(cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True); self.timeout = timeout; self.max_workers = max(1, int(max_workers))
        self._captures = {}; self._capture_diag = {}; self._snapshot_diag = {}

    @staticmethod
    def _row_key(row): return _row_key(row)
    @staticmethod
    def _cache_key(value): return hashlib.sha256(value.encode()).hexdigest()

    def capture_diagnostic(self, url):
        self.captures(url); return self._capture_diag.get(url, CaptureDiagnostic("CDX_REQUEST_FAILURE", error_type="UNKNOWN"))

    def captures(self, url):
        if url in self._captures: return self._captures[url]
        cache = self.cache_dir / f"captures_{self._cache_key(url)}.json"
        if cache.exists():
            try:
                rows = json.loads(cache.read_text(encoding="utf-8"));
                if not isinstance(rows, list): raise ValueError("capture cache is not a list")
                self._captures[url] = rows; self._capture_diag[url] = CaptureDiagnostic("CDX_CAPTURE_FOUND" if rows else "CDX_NO_CAPTURE", len(rows)); return rows
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                self._capture_diag[url] = CaptureDiagnostic("CDX_CACHE_FAILURE", error_type=type(exc).__name__, error=str(exc))
        params = {"url": url, "output": "json", "filter": "statuscode:200", "fl": "timestamp,digest,original,statuscode,mimetype"}
        try:
            response = requests.get(WAYBACK_CDX, params=params, timeout=self.timeout, headers={"User-Agent": "SoccerPredictionResearch/1.0 PIT-Audit"}); response.raise_for_status(); payload = response.json()
        except requests.RequestException as exc:
            self._captures[url] = []; self._capture_diag[url] = CaptureDiagnostic("CDX_REQUEST_FAILURE", error_type=type(exc).__name__, error=str(exc)); return []
        except (ValueError, TypeError) as exc:
            self._captures[url] = []; self._capture_diag[url] = CaptureDiagnostic("CDX_RESPONSE_PARSE_FAILURE", error_type=type(exc).__name__, error=str(exc)); return []
        rows = [] if not payload or len(payload) <= 1 else [dict(zip(payload[0], row)) for row in payload[1:]]; rows.sort(key=lambda r: r.get("timestamp", ""))
        try: cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        except OSError: self._capture_diag[url] = CaptureDiagnostic("CDX_CACHE_FAILURE", len(rows), error_type="OSError", error="capture cache write failed")
        self._captures[url] = rows
        if url not in self._capture_diag or self._capture_diag[url].status != "CDX_CACHE_FAILURE": self._capture_diag[url] = CaptureDiagnostic("CDX_CAPTURE_FOUND" if rows else "CDX_NO_CAPTURE", len(rows))
        return rows

    @staticmethod
    def _snapshot_url(capture, original_url): return f"{WAYBACK_WEB}/{capture['timestamp']}id_/{original_url}"
    def _snapshot_cache_path(self, capture):
        identity = f"{capture.get('digest','')}|{capture.get('timestamp','')}|{capture.get('original','')}"; return self.cache_dir / f"snapshot_{self._cache_key(identity)}.csv"

    def _load_snapshot_keys(self, capture, original_url):
        identity = f"{capture.get('digest','')}|{capture.get('timestamp','')}|{capture.get('original','')}"
        if identity in self._snapshot_diag: return self._snapshot_diag[identity].keys or set()
        cache = self._snapshot_cache_path(capture)
        try:
            if cache.exists(): raw = cache.read_bytes()
            else:
                response = requests.get(self._snapshot_url(capture, original_url), timeout=self.timeout, headers={"User-Agent": "SoccerPredictionResearch/1.0 PIT-Audit"}); response.raise_for_status(); raw = response.content; cache.write_bytes(raw)
            frame = pd.read_csv(BytesIO(raw)); keys = {_row_key(row) for _, row in frame.iterrows()}; keys.discard(None)
            self._snapshot_diag[identity] = SnapshotDiagnostic("SNAPSHOT_PARSED", keys=keys); return keys
        except requests.RequestException as exc:
            self._snapshot_diag[identity] = SnapshotDiagnostic("SNAPSHOT_REQUEST_FAILURE", error_type=type(exc).__name__, error=str(exc))
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            self._snapshot_diag[identity] = SnapshotDiagnostic("SNAPSHOT_PARSE_FAILURE", error_type=type(exc).__name__, error=str(exc))
        return set()


def competition_adapter_matrix() -> pd.DataFrame:
    rows = []
    for competition, spec in COMPETITION_ADAPTERS.items():
        rows.append({"competition": competition, "source": spec.get("source"), "source_code": spec.get("source_code"), "adapter": spec.get("adapter"), "status": "IMPLEMENTED" if spec.get("adapter") else "UNVERIFIED"})
    return pd.DataFrame(rows)
