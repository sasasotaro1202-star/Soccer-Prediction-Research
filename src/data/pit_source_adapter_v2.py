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
INPUT_TO_FIXED = {"EPL": "E0", "CHA": "CH", "BL1": "D1", "SA": "I1", "SP1": "SP1", "LL": "SP1", "FL1": "F1", "ERE": "N1"}

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
    if value is None or value == "": return None
    try:
        if pd.isna(value): return None
    except (TypeError, ValueError): pass
    text = str(value).strip()
    if not text: return None
    if len(text) == 14 and text.isdigit():
        try: return datetime.strptime(text, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError: return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError): return None


def source_url(competition: str, start_year: int) -> str:
    fixed = INPUT_TO_FIXED.get(competition, competition); spec = COMPETITION_ADAPTERS.get(fixed)
    if not spec or not spec["source_code"]: raise ValueError(f"No PIT source mapping for {competition}")
    return BASE.format(season_folder=season_folder(start_year), league=spec["source_code"])


def _date_key(value: Any) -> str | None:
    """Normalize dates deterministically; ISO YYYY-MM-DD is never dayfirst-parsed."""
    if value is None or value == "": return None
    try:
        if pd.isna(value): return None
    except (TypeError, ValueError): pass
    text = str(value).strip()
    if not text: return None
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        iso_text = text.replace("Z", "+00:00")
        try: return datetime.fromisoformat(iso_text).date().isoformat()
        except ValueError:
            try: return datetime.strptime(text[:10], "%Y-%m-%d").date().isoformat()
            except ValueError: pass
    for fmt in ("%d/%m/%y", "%d/%m/%Y", "%d-%m-%y", "%d-%m-%Y"):
        try: return datetime.strptime(text, fmt).date().isoformat()
        except ValueError: pass
    parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    return None if pd.isna(parsed) else parsed.date().isoformat()


def _source_date_key(row: pd.Series) -> str | None:
    key = _date_key(row.get("source_event_date"))
    return key if key is not None else _date_key(row.get("kickoff_utc"))


def _row_key(row: pd.Series) -> tuple | None:
    date = _source_date_key(row)
    if date is None: return None
    try: return (date, str(row.get("home_team", "")).strip(), str(row.get("away_team", "")).strip(), float(row.get("home_goals")), float(row.get("away_goals")), str(row.get("result", "")).strip())
    except (TypeError, ValueError): return None


def _result_lower_bound(row: pd.Series) -> tuple[datetime | None, str]:
    kickoff = _utc(row.get("kickoff_utc"))
    if kickoff is None: return None, "MISSING_EVENT_TIME"
    if bool(row.get("kickoff_time_available", False)): return kickoff + timedelta(minutes=180), "KICKOFF_PLUS_180M"
    return kickoff.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1), "DATE_ONLY_NEXT_DAY"


class FootballDataWaybackAdapter:
    def __init__(self, cache_dir: str = "data/raw/pit_evidence", timeout: int = 30, max_workers: int = DEFAULT_MAX_WORKERS):
        self.cache_dir = Path(cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True); self.timeout = timeout; self.max_workers = max(1, int(max_workers))
        self._captures: dict[str, list[dict[str, str]]] = {}; self._capture_diag: dict[str, CaptureDiagnostic] = {}; self._snapshot_diag: dict[str, SnapshotDiagnostic] = {}
    @staticmethod
    def _row_key(row: pd.Series) -> tuple | None: return _row_key(row)
    @staticmethod
    def _cache_key(value: str) -> str: return hashlib.sha256(value.encode()).hexdigest()
    def capture_diagnostic(self, url: str) -> CaptureDiagnostic:
        self.captures(url); return self._capture_diag.get(url, CaptureDiagnostic("CDX_REQUEST_FAILURE", error_type="UNKNOWN"))
    def captures(self, url: str) -> list[dict[str, str]]:
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
    def _snapshot_url(capture: dict[str, str], original_url: str) -> str: return f"{WAYBACK_WEB}/{capture['timestamp']}id_/{original_url}"
    def _snapshot_cache_path(self, capture: dict[str, str]) -> Path:
        identity = f"{capture.get('digest','')}|{capture.get('timestamp','')}|{capture.get('original','')}"; return self.cache_dir / f"snapshot_{self._cache_key(identity)}.csv"
    def _load_snapshot_keys(self, capture: dict[str, str], original_url: str) -> SnapshotDiagnostic:
        identity = f"{capture.get('digest','')}|{capture.get('timestamp','')}|{capture.get('original','')}"
        if identity in self._snapshot_diag: return self._snapshot_diag[identity]
        cache = self._snapshot_cache_path(capture)
        try:
            raw = cache.read_bytes() if cache.exists() else None
            if raw is None:
                response = requests.get(self._snapshot_url(capture, original_url), timeout=self.timeout, headers={"User-Agent": "SoccerPredictionResearch/1.0 PIT-Audit"}); response.raise_for_status(); raw = response.content; cache.write_bytes(raw)
        except requests.RequestException as exc:
            diag = SnapshotDiagnostic("SNAPSHOT_DOWNLOAD_FAILURE", error_type=type(exc).__name__, error=str(exc)); self._snapshot_diag[identity] = diag; return diag
        except OSError as exc:
            diag = SnapshotDiagnostic("SNAPSHOT_CACHE_FAILURE", error_type=type(exc).__name__, error=str(exc)); self._snapshot_diag[identity] = diag; return diag
        try: frame = pd.read_csv(BytesIO(raw))
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            diag = SnapshotDiagnostic("SNAPSHOT_PARSE_FAILURE", error_type=type(exc).__name__, error=str(exc)); self._snapshot_diag[identity] = diag; return diag
        required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
        if not required.issubset(frame.columns):
            diag = SnapshotDiagnostic("SNAPSHOT_SCHEMA_FAILURE", error_type="MissingColumns", error=",".join(sorted(required-set(frame.columns)))); self._snapshot_diag[identity] = diag; return diag
        keys = set()
        for r in frame.itertuples(index=False):
            date = _date_key(getattr(r, "Date", None))
            if date is None: continue
            try: keys.add((date, str(getattr(r, "HomeTeam")).strip(), str(getattr(r, "AwayTeam")).strip(), float(getattr(r, "FTHG")), float(getattr(r, "FTAG")), str(getattr(r, "FTR")).strip()))
            except (TypeError, ValueError): continue
        diag = SnapshotDiagnostic("SNAPSHOT_PARSED", keys=keys); self._snapshot_diag[identity] = diag; return diag
    def _prefetch_url(self, url: str, rows: list[pd.Series], workers: int | None = None) -> list[SourceEvidence]:
        captures = self.captures(url)
        if not captures:
            diag = self._capture_diag.get(url, CaptureDiagnostic("CDX_REQUEST_FAILURE")); reason = "no_archive_captures" if diag.status == "CDX_NO_CAPTURE" else f"{diag.status.lower()}: {diag.error or ''}".strip(); return [SourceEvidence(None, "UNVERIFIABLE", reason=reason) for _ in rows]
        row_keys = [_row_key(row) for row in rows]; bounds = [_result_lower_bound(row) for row in rows]
        min_bound = min((lb for lb, _ in bounds if lb is not None), default=None)
        kickoff_bounds = [_utc(row.get("kickoff_utc")) if bool(row.get("kickoff_time_available", False)) else None for row in rows]
        min_search_bound = min((x for x in kickoff_bounds if x is not None), default=min_bound)
        candidates = [c for c in captures if (_utc(c.get("timestamp")) is not None and min_search_bound is not None and _utc(c.get("timestamp")) >= min_search_bound)]
        candidates.sort(key=lambda c:c.get("timestamp", ""))
        if not candidates:
            return [SourceEvidence(None, "UNVERIFIABLE", reason=f"captures_exist_but_no_capture_after_result_lower_bound:{bound_reason}") for _, bound_reason in bounds]
        def fetch(c): return c, self._load_snapshot_keys(c, url)
        keysets=[]
        with ThreadPoolExecutor(max_workers=workers or self.max_workers) as pool:
            for f in as_completed([pool.submit(fetch,c) for c in candidates]): keysets.append(f.result())
        keysets.sort(key=lambda x:x[0].get("timestamp", ""))
        results=[None]*len(rows)
        unresolved=set(i for i,k in enumerate(row_keys) if k is not None)
        # First pass: conservative result-completion bound.
        for capture,diag in keysets:
            if not unresolved: break
            ts=_utc(capture.get("timestamp"))
            if ts is None or diag.keys is None: continue
            for i in list(unresolved):
                lb,bound_reason=bounds[i]
                if lb is not None and ts >= lb and row_keys[i] in diag.keys:
                    results[i]=SourceEvidence(ts.isoformat(),"VERIFIED",self._snapshot_url(capture,url),capture.get("digest"),f"archived_completed_result_first_observed_after_{bound_reason.lower()}")
                    unresolved.remove(i)
        # Second pass: when exact kickoff time is known, an archive observed after kickoff
        # is valid evidence that the completed-result row was published by that capture.
        # This is intentionally narrower than the conservative bound and never accepts
        # a capture before kickoff.
        for capture,diag in keysets:
            if not unresolved: break
            ts=_utc(capture.get("timestamp"))
            if ts is None or diag.keys is None: continue
            for i in list(unresolved):
                kickoff=kickoff_bounds[i]
                if kickoff is not None and kickoff <= ts and row_keys[i] in diag.keys:
                    results[i]=SourceEvidence(ts.isoformat(),"VERIFIED",self._snapshot_url(capture,url),capture.get("digest"),"archived_completed_result_first_observed_after_kickoff")
                    unresolved.remove(i)
        out=[]
        for i,result in enumerate(results):
            if result is not None: out.append(result); continue
            key=row_keys[i]; lb,_=bounds[i]
            out.append(SourceEvidence(None,"UNVERIFIABLE",reason="missing_record_identity" if key is None else ("missing_event_time" if lb is None else "no_archive_snapshot_contains_completed_result")))
        return out
    def apply_bulk(self, history: pd.DataFrame) -> pd.DataFrame:
        if history.empty: return history.copy()
        out=history.copy(); out["source_available_at_utc"]=None; out["pit_evidence_status"]="UNVERIFIABLE"; out["pit_evidence_reason"]=""; out["pit_evidence_url"]=None; out["capture_digest"]=None
        grouped={}
        for idx,row in out.iterrows():
            try: grouped.setdefault((str(row["competition"]),int(row["season_start"])),[]).append((idx,row))
            except (KeyError,TypeError,ValueError): out.at[idx,"pit_evidence_reason"]="missing_competition_or_season"
        for (competition,start_year),items in grouped.items():
            try: url=source_url(competition,start_year); evidences=self._prefetch_url(url,[r for _,r in items])
            except Exception as exc: evidences=[SourceEvidence(None,"UNVERIFIABLE",reason=f"adapter_failure:{type(exc).__name__}:{exc}") for _ in items]
            for (idx,_),ev in zip(items,evidences):
                out.at[idx,"source_available_at_utc"]=ev.source_available_at_utc; out.at[idx,"pit_evidence_status"]=ev.evidence_status; out.at[idx,"pit_evidence_reason"]=ev.reason; out.at[idx,"pit_evidence_url"]=ev.evidence_url; out.at[idx,"capture_digest"]=ev.capture_digest
        return out
