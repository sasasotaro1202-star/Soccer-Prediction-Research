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
    "UCL": {"source": "UNVERIFIED", "source_code": None, "adapter": None},
    "UEL": {"source": "UNVERIFIED", "source_code": None, "adapter": None},
    "J1": {"source": "UNVERIFIED", "source_code": None, "adapter": None},
    "J2": {"source": "UNVERIFIED", "source_code": None, "adapter": None},
    "J3": {"source": "UNVERIFIED", "source_code": None, "adapter": None},
    "DFBP": {"source": "UNVERIFIED", "source_code": None, "adapter": None},
    "FRIENDLY": {"source": "UNVERIFIED", "source_code": None, "adapter": None},
    "EFL": {"source": "UNVERIFIED", "source_code": None, "adapter": None},
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
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return None
    if len(text) == 14 and text.isdigit():
        try:
            return datetime.strptime(text, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def source_url(competition: str, start_year: int) -> str:
    fixed = INPUT_TO_FIXED.get(competition, competition)
    spec = COMPETITION_ADAPTERS.get(fixed)
    if not spec or not spec["source_code"]:
        raise ValueError(f"No PIT source mapping for {competition}")
    return BASE.format(season_folder=season_folder(start_year), league=spec["source_code"])


def _date_key(value: Any) -> str | None:
    """Normalize source dates without reordering ISO dates."""
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return None
    # Football-Data uses DD/MM/YY, while PIT fallback can supply ISO timestamps.
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return parsed.date().isoformat()
        except ValueError:
            try:
                return datetime.strptime(text[:10], "%Y-%m-%d").date().isoformat()
            except ValueError:
                pass
    for fmt in ("%d/%m/%y", "%d/%m/%Y", "%d-%m-%y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    return None if pd.isna(parsed) else parsed.date().isoformat()


def _source_date_key(row: pd.Series) -> str | None:
    key = _date_key(row.get("source_event_date"))
    if key is not None:
        return key
    return _date_key(row.get("kickoff_utc"))


def _row_key(row: pd.Series) -> tuple | None:
    date = _source_date_key(row)
    if date is None:
        return None
    try:
        return (date, str(row.get("home_team", "")).strip(), str(row.get("away_team", "")).strip(), float(row.get("home_goals")), float(row.get("away_goals")), str(row.get("result", "")).strip())
    except (TypeError, ValueError):
        return None


def _result_lower_bound(row: pd.Series) -> tuple[datetime | None, str]:
    kickoff = _utc(row.get("kickoff_utc"))
    if kickoff is None:
        return None, "MISSING_EVENT_TIME"
    if bool(row.get("kickoff_time_available", False)):
        return kickoff + timedelta(minutes=180), "KICKOFF_PLUS_180M"
    return kickoff.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1), "DATE_ONLY_NEXT_DAY"


class FootballDataWaybackAdapter:
    def __init__(self, cache_dir: str = "data/raw/pit_evidence", timeout: int = 30, max_workers: int = DEFAULT_MAX_WORKERS):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_workers = max(1, int(max_workers))
        self._captures: dict[str, list[dict[str, str]]] = {}
        self._capture_diag: dict[str, CaptureDiagnostic] = {}
        self._snapshot_diag: dict[str, SnapshotDiagnostic] = {}

    @staticmethod
    def _row_key(row: pd.Series) -> tuple | None:
        return _row_key(row)

    @staticmethod
    def _cache_key(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    def capture_diagnostic(self, url: str) -> CaptureDiagnostic:
        self.captures(url)
        return self._capture_diag.get(url, CaptureDiagnostic("CDX_REQUEST_FAILURE", error_type="UNKNOWN"))

    def captures(self, url: str) -> list[dict[str, str]]:
        if url in self._captures:
            return self._captures[url]
        cache = self.cache_dir / f"captures_{self._cache_key(url)}.json"
        if cache.exists():
            try:
                rows = json.loads(cache.read_text(encoding="utf-8"))
                if not isinstance(rows, list):
                    raise ValueError("capture cache is not a list")
                self._captures[url] = rows
                self._capture_diag[url] = CaptureDiagnostic("CDX_CAPTURE_FOUND" if rows else "CDX_NO_CAPTURE", len(rows))
                return rows
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                self._capture_diag[url] = CaptureDiagnostic("CDX_CACHE_FAILURE", error_type=type(exc).__name__, error=str(exc))
        params = {"url": url, "output": "json", "filter": "statuscode:200", "fl": "timestamp,digest,original,statuscode,mimetype"}
        try:
            response = requests.get(WAYBACK_CDX, params=params, timeout=self.timeout, headers={"User-Agent": "SoccerPredictionResearch/1.0 PIT-Audit"})
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            self._captures[url] = []
            self._capture_diag[url] = CaptureDiagnostic("CDX_REQUEST_FAILURE", error_type=type(exc).__name__, error=str(exc))
            return []
        except (ValueError, TypeError) as exc:
            self._captures[url] = []
            self._capture_diag[url] = CaptureDiagnostic("CDX_RESPONSE_PARSE_FAILURE", error_type=type(exc).__name__, error=str(exc))
            return []
        rows = [] if not payload or len(payload) <= 1 else [dict(zip(payload[0], row)) for row in payload[1:]]
        rows.sort(key=lambda r: r.get("timestamp", ""))
        try:
            cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        except OSError:
            self._capture_diag[url] = CaptureDiagnostic("CDX_CACHE_FAILURE", len(rows), error_type="OSError", error="capture cache write failed")
        self._captures[url] = rows
        if url not in self._capture_diag or self._capture_diag[url].status != "CDX_CACHE_FAILURE":
            self._capture_diag[url] = CaptureDiagnostic("CDX_CAPTURE_FOUND" if rows else "CDX_NO_CAPTURE", len(rows))
        return rows

    @staticmethod
    def _snapshot_url(capture: dict[str, str], original_url: str) -> str:
        return f"{WAYBACK_WEB}/{capture['timestamp']}id_/{original_url}"

    def _snapshot_cache_path(self, capture: dict[str, str]) -> Path:
        identity = f"{capture.get('digest','')}|{capture.get('timestamp','')}|{capture.get('original','')}"
        return self.cache_dir / f"snapshot_{self._cache_key(identity)}.csv"

    def _load_snapshot_keys(self, capture: dict[str, str], original_url: str) -> SnapshotDiagnostic:
        identity = f"{capture.get('digest','')}|{capture.get('timestamp','')}|{capture.get('original','')}"
        if identity in self._snapshot_diag:
            return self._snapshot_diag[identity]
        cache = self._snapshot_cache_path(capture)
        try:
            raw = cache.read_bytes() if cache.exists() else None
            if raw is None:
                response = requests.get(self._snapshot_url(capture, original_url), timeout=self.timeout, headers={"User-Agent": "SoccerPredictionResearch/1.0 PIT-Audit"})
                response.raise_for_status()
                raw = response.content
                cache.write_bytes(raw)
        except requests.RequestException as exc:
            diag = SnapshotDiagnostic("SNAPSHOT_DOWNLOAD_FAILURE", error_type=type(exc).__name__, error=str(exc))
            self._snapshot_diag[identity] = diag
            return diag
        except OSError as exc:
            diag = SnapshotDiagnostic("SNAPSHOT_CACHE_FAILURE", error_type=type(exc).__name__, error=str(exc))
            self._snapshot_diag[identity] = diag
            return diag
        try:
            frame = pd.read_csv(BytesIO(raw))
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            diag = SnapshotDiagnostic("SNAPSHOT_PARSE_FAILURE", error_type=type(exc).__name__, error=str(exc))
            self._snapshot_diag[identity] = diag
            return diag
        required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
        if not required.issubset(frame.columns):
            missing = ",".join(sorted(required - set(frame.columns)))
            diag = SnapshotDiagnostic("SNAPSHOT_SCHEMA_FAILURE", error_type="MissingColumns", error=missing)
            self._snapshot_diag[identity] = diag
            return diag
        keys: set[tuple] = set()
        for r in frame.itertuples(index=False):
            date = _date_key(getattr(r, "Date", None))
            if date is None:
                continue
            try:
                keys.add((date, str(getattr(r, "HomeTeam")).strip(), str(getattr(r, "AwayTeam")).strip(), float(getattr(r, "FTHG")), float(getattr(r, "FTAG")), str(getattr(r, "FTR")).strip()))
            except (TypeError, ValueError):
                continue
        diag = SnapshotDiagnostic("SNAPSHOT_PARSED", keys=keys)
        self._snapshot_diag[identity] = diag
        return diag

    def _prefetch_url(self, url: str, rows: list[pd.Series], workers: int | None = None) -> list[SourceEvidence]:
        captures = self.captures(url)
        if not captures:
            diag = self._capture_diag.get(url, CaptureDiagnostic("CDX_REQUEST_FAILURE"))
            reason = "no_archive_captures" if diag.status == "CDX_NO_CAPTURE" else f"{diag.status.lower()}: {diag.error or ''}".strip()
            return [SourceEvidence(None, "UNVERIFIABLE", reason=reason) for _ in rows]
        row_keys = [_row_key(row) for row in rows]
        bounds = [_result_lower_bound(row) for row in rows]
        candidates = []
        for capture in captures:
            ts = _utc(capture.get("timestamp"))
            if ts is not None and any(lb is not None and ts >= lb for lb, _ in bounds):
                candidates.append(capture)
        candidates.sort(key=lambda c: c.get("timestamp", ""))
        if not candidates:
            return [SourceEvidence(None, "UNVERIFIABLE", reason=f"captures_exist_but_no_capture_after_result_lower_bound:{bound_reason}") for _, bound_reason in bounds]
        def fetch(capture):
            return capture, self._load_snapshot_keys(capture, url)
        keysets = []
        with ThreadPoolExecutor(max_workers=workers or self.max_workers) as pool:
            futures = [pool.submit(fetch, c) for c in candidates]
            for future in as_completed(futures):
                keysets.append(future.result())
        keysets.sort(key=lambda x: x[0].get("timestamp", ""))
        results = []
        for key, (lb, bound_reason) in zip(row_keys, bounds):
            if key is None or lb is None:
                results.append(SourceEvidence(None, "UNVERIFIABLE", reason="missing_record_identity"))
                continue
            best = None
            snapshot_errors = []
            for capture, diag in keysets:
                ts = _utc(capture.get("timestamp"))
                if ts is None or ts < lb:
                    continue
                if diag.keys is None:
                    snapshot_errors.append(diag.status)
                    continue
                if key in diag.keys:
                    best = (ts, capture)
                    break
            if best is None:
                reason = "no_archive_snapshot_contains_completed_result"
                if snapshot_errors:
                    reason += ":" + ",".join(sorted(set(snapshot_errors)))
                results.append(SourceEvidence(None, "UNVERIFIABLE", reason=reason))
            else:
                ts, capture = best
                results.append(SourceEvidence(ts.isoformat(), "VERIFIED", self._snapshot_url(capture, url), capture.get("digest"), f"archived_completed_result_first_observed_after_{bound_reason.lower()}"))
        return results

    def apply_bulk(self, history: pd.DataFrame) -> pd.DataFrame:
        if history.empty:
            return history.copy()
        out = history.copy()
        groups: dict[str, list[tuple[int, pd.Series]]] = {}
        evidence: dict[int, SourceEvidence] = {}
        for idx, row in out.iterrows():
            try:
                year = int(str(row.get("season", "0000/00")).split("/")[0])
                groups.setdefault(source_url(str(row.get("competition", "")), year), []).append((idx, row))
            except (ValueError, TypeError) as exc:
                evidence[idx] = SourceEvidence(None, "UNVERIFIABLE", reason=str(exc))
        def run_group(item):
            url, indexed = item
            return indexed, self._prefetch_url(url, [row for _, row in indexed], workers=1)
        with ThreadPoolExecutor(max_workers=min(self.max_workers, max(1, len(groups)))) as pool:
            futures = [pool.submit(run_group, item) for item in groups.items()]
            for future in as_completed(futures):
                indexed, results = future.result()
                for (idx, _), result in zip(indexed, results):
                    evidence[idx] = result
        out["source_available_at_utc"] = [evidence[i].source_available_at_utc for i in out.index]
        out["pit_evidence_status"] = [evidence[i].evidence_status for i in out.index]
        out["pit_evidence_url"] = [evidence[i].evidence_url for i in out.index]
        out["pit_evidence_digest"] = [evidence[i].capture_digest for i in out.index]
        out["pit_evidence_reason"] = [evidence[i].reason for i in out.index]
        return out

    def evidence_for_row(self, row: pd.Series) -> SourceEvidence:
        year = int(str(row.get("season", "0000/00")).split("/")[0])
        return self._prefetch_url(source_url(str(row.get("competition", "")), year), [row], workers=self.max_workers)[0]

    def diagnostic_bulk(self, history: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for (competition, season), group in history.groupby(["competition", "season"], dropna=False):
            try:
                year = int(str(season).split("/")[0])
                url = source_url(str(competition), year)
            except Exception as exc:
                rows.append({"competition": competition, "season": season, "source": "UNVERIFIED", "cdx_status": "SOURCE_MAPPING_FAILURE", "cdx_capture_count": 0, "first_capture_at": None, "last_capture_at": None, "captures_after_result_lower_bound": 0, "snapshot_attempt_count": 0, "snapshot_success_count": 0, "result_match_count": 0, "pit_verified_count": 0, "failure_stage": "SOURCE_MAPPING_FAILURE", "failure_reason": str(exc)})
                continue
            captures = self.captures(url)
            cdiag = self._capture_diag.get(url, CaptureDiagnostic("CDX_REQUEST_FAILURE"))
            times = [t for t in (_utc(c.get("timestamp")) for c in captures) if t is not None]
            lower_bounds = [_result_lower_bound(r)[0] for _, r in group.iterrows()]
            eligible = [c for c in captures if (_utc(c.get("timestamp")) is not None and any(lb is not None and _utc(c.get("timestamp")) >= lb for lb in lower_bounds))]
            snapshot_attempt_count = snapshot_success_count = result_match_count = 0
            for capture in eligible:
                snapshot_attempt_count += 1
                diag = self._load_snapshot_keys(capture, url)
                if diag.keys is not None:
                    snapshot_success_count += 1
                    result_match_count += sum(1 for _, r in group.iterrows() if (_row_key(r) in diag.keys if _row_key(r) is not None else False))
            rows.append({"competition": competition, "season": season, "source": "Football-Data.co.uk", "cdx_status": cdiag.status, "cdx_capture_count": len(captures), "first_capture_at": min(times).isoformat() if times else None, "last_capture_at": max(times).isoformat() if times else None, "captures_after_result_lower_bound": len(eligible), "snapshot_attempt_count": snapshot_attempt_count, "snapshot_success_count": snapshot_success_count, "result_match_count": result_match_count, "pit_verified_count": result_match_count, "failure_stage": "OK" if result_match_count else cdiag.status if not captures else "NO_RESULT_MATCH", "failure_reason": cdiag.error or ("no eligible snapshot" if not eligible else "no matching completed result")})
        return pd.DataFrame(rows)


def apply_pit_evidence(history: pd.DataFrame) -> pd.DataFrame:
    return FootballDataWaybackAdapter().apply_bulk(history)


def build_pit_diagnostic(history: pd.DataFrame) -> pd.DataFrame:
    return FootballDataWaybackAdapter().diagnostic_bulk(history)


def competition_adapter_matrix() -> pd.DataFrame:
    rows = []
    for competition, spec in COMPETITION_ADAPTERS.items():
        rows.append({"competition": competition, "source": spec["source"], "source_code": spec["source_code"], "adapter": spec["adapter"], "status": "IMPLEMENTED" if spec["adapter"] else "UNVERIFIED"})
    return pd.DataFrame(rows)
