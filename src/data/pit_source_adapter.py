from __future__ import annotations

"""Bulk-first PIT archive adapter with bounded parallel snapshot I/O."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    fixed = INPUT_TO_FIXED.get(competition, competition)
    spec = COMPETITION_ADAPTERS.get(fixed)
    if not spec or not spec["source_code"]:
        raise ValueError(f"No PIT source mapping for {competition}")
    return BASE.format(season_folder=season_folder(start_year), league=spec["source_code"])

def _date_key(value: Any) -> str | None:
    parsed = pd.to_datetime(value, dayfirst=True, errors="coerce")
    return None if pd.isna(parsed) else parsed.date().isoformat()

def _row_key(row: pd.Series) -> tuple | None:
    kickoff = _utc(row.get("kickoff_utc"))
    if kickoff is None:
        return None
    try:
        return (kickoff.date().isoformat(), str(row.get("home_team", "")).strip(), str(row.get("away_team", "")).strip(), float(row.get("home_goals")), float(row.get("away_goals")), str(row.get("result", "")).strip())
    except (TypeError, ValueError):
        return None

class FootballDataWaybackAdapter:
    """Fetch capture metadata once, cache snapshots, and verify rows locally."""
    def __init__(self, cache_dir: str = "data/raw/pit_evidence", timeout: int = 30, max_workers: int = DEFAULT_MAX_WORKERS):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_workers = max(1, int(max_workers))
        self._captures: dict[str, list[dict[str, str]]] = {}
        self._snapshot_keys: dict[str, set[tuple]] = {}

    @staticmethod
    def _row_key(row: pd.Series) -> tuple | None:
        return _row_key(row)

    @staticmethod
    def _cache_key(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    def captures(self, url: str) -> list[dict[str, str]]:
        if url in self._captures:
            return self._captures[url]
        cache = self.cache_dir / f"captures_{self._cache_key(url)}.json"
        if cache.exists():
            try:
                rows = json.loads(cache.read_text(encoding="utf-8"))
                self._captures[url] = rows
                return rows
            except (OSError, json.JSONDecodeError):
                pass
        params = {"url": url, "output": "json", "filter": "statuscode:200", "fl": "timestamp,digest,original,statuscode,mimetype", "collapse": "digest"}
        try:
            r = requests.get(WAYBACK_CDX, params=params, timeout=self.timeout, headers={"User-Agent": "SoccerPredictionResearch/1.0 PIT-Audit"})
            r.raise_for_status()
            payload = r.json()
        except Exception:
            self._captures[url] = []
            return []
        rows = [] if not payload or len(payload) <= 1 else [dict(zip(payload[0], row)) for row in payload[1:]]
        try:
            cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
        self._captures[url] = rows
        return rows

    @staticmethod
    def _snapshot_url(capture: dict[str, str], original_url: str) -> str:
        return f"{WAYBACK_WEB}/{capture['timestamp']}id_/{original_url}"

    def _snapshot_cache_path(self, capture: dict[str, str]) -> Path:
        identity = capture.get("digest") or capture.get("timestamp", "")
        return self.cache_dir / f"snapshot_{self._cache_key(identity)}.csv"

    def _snapshot_keyset(self, capture: dict[str, str], original_url: str) -> set[tuple] | None:
        digest = capture.get("digest") or capture.get("timestamp", "")
        if digest in self._snapshot_keys:
            return self._snapshot_keys[digest]
        cache = self._snapshot_cache_path(capture)
        try:
            raw = cache.read_bytes() if cache.exists() else None
            if raw is None:
                r = requests.get(self._snapshot_url(capture, original_url), timeout=self.timeout, headers={"User-Agent": "SoccerPredictionResearch/1.0 PIT-Audit"})
                r.raise_for_status()
                raw = r.content
                cache.write_bytes(raw)
            frame = pd.read_csv(BytesIO(raw))
        except Exception:
            return None
        required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
        if not required.issubset(frame.columns):
            return None
        keys: set[tuple] = set()
        for r in frame.itertuples(index=False):
            date = _date_key(getattr(r, "Date", None))
            if date is None:
                continue
            try:
                keys.add((date, str(getattr(r, "HomeTeam")).strip(), str(getattr(r, "AwayTeam")).strip(), float(getattr(r, "FTHG")), float(getattr(r, "FTAG")), str(getattr(r, "FTR")).strip()))
            except (TypeError, ValueError):
                continue
        self._snapshot_keys[digest] = keys
        return keys

    def _prefetch_url(self, url: str, rows: list[pd.Series], workers: int | None = None) -> list[SourceEvidence]:
        captures = self.captures(url)
        if not captures:
            return [SourceEvidence(None, "UNVERIFIABLE", reason="no_archive_captures") for _ in rows]
        row_keys = [_row_key(row) for row in rows]
        lower_bounds = []
        for row in rows:
            event = _utc(row.get("kickoff_utc"))
            lower_bounds.append(event.replace(hour=23, minute=59, second=59, microsecond=999999) if event else None)
        candidates: dict[str, tuple[datetime, dict[str, str]]] = {}
        for capture in captures:
            ts = _utc(capture.get("timestamp"))
            digest = capture.get("digest") or capture.get("timestamp", "")
            if ts is not None and any(lb is not None and ts >= lb for lb in lower_bounds):
                candidates[digest] = (ts, capture)

        def fetch(item):
            digest, (_, capture) = item
            return digest, self._snapshot_keyset(capture, url)

        with ThreadPoolExecutor(max_workers=workers or self.max_workers) as pool:
            future_map = [pool.submit(fetch, item) for item in candidates.items()]
            keysets = {}
            for future in as_completed(future_map):
                digest, keys = future.result()
                keysets[digest] = keys

        results = []
        for key, lb in zip(row_keys, lower_bounds):
            if key is None or lb is None:
                results.append(SourceEvidence(None, "UNVERIFIABLE", reason="missing_record_identity"))
                continue
            best = None
            for digest, (ts, capture) in candidates.items():
                if ts >= lb and keysets.get(digest) is not None and key in keysets[digest] and (best is None or ts < best[0]):
                    best = (ts, capture)
            if best is None:
                results.append(SourceEvidence(None, "UNVERIFIABLE", reason="no_archive_snapshot_contains_completed_result"))
            else:
                ts, capture = best
                results.append(SourceEvidence(ts.isoformat(), "VERIFIED", self._snapshot_url(capture, url), capture.get("digest"), "archived_completed_result_first_observed_after_event_date"))
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
        """Return auditable row-level failure-stage telemetry without inventing PIT timestamps."""
        columns = ["competition", "season", "source", "field", "rows", "cdx_capture_count", "snapshot_attempt_count", "snapshot_success_count", "parse_success_count", "match_key_match_count", "result_match_count", "source_available_at_count", "pit_verified_count", "failure_stage", "failure_reason"]
        if history.empty:
            return pd.DataFrame(columns=columns)

        out_rows: list[dict[str, Any]] = []
        grouped = history.groupby(["competition", "season"], dropna=False, sort=True)
        for (competition, season), group in grouped:
            try:
                year = int(str(season).split("/")[0])
                url = source_url(str(competition), year)
                source = COMPETITION_ADAPTERS[INPUT_TO_FIXED.get(str(competition), str(competition))]["source"]
            except Exception as exc:
                out_rows.append({"competition": competition, "season": season, "source": "UNVERIFIED", "field": "completed_result", "rows": len(group), "cdx_capture_count": 0, "snapshot_attempt_count": 0, "snapshot_success_count": 0, "parse_success_count": 0, "match_key_match_count": 0, "result_match_count": 0, "source_available_at_count": 0, "pit_verified_count": 0, "failure_stage": "CDX_CAPTURE_FAILURE", "failure_reason": str(exc)})
                continue

            captures = self.captures(url)
            cdx_count = len(captures)
            if not captures:
                out_rows.append({"competition": competition, "season": season, "source": source, "field": "completed_result", "rows": len(group), "cdx_capture_count": 0, "snapshot_attempt_count": 0, "snapshot_success_count": 0, "parse_success_count": 0, "match_key_match_count": 0, "result_match_count": 0, "source_available_at_count": 0, "pit_verified_count": 0, "failure_stage": "CDX_CAPTURE_FAILURE", "failure_reason": "no_archive_captures"})
                continue

            bounds = []
            for _, row in group.iterrows():
                event = _utc(row.get("kickoff_utc"))
                bounds.append(event.replace(hour=23, minute=59, second=59, microsecond=999999) if event else None)
            candidates = {}
            for capture in captures:
                ts = _utc(capture.get("timestamp"))
                digest = capture.get("digest") or capture.get("timestamp", "")
                if ts is not None and any(lb is not None and ts >= lb for lb in bounds):
                    candidates[digest] = (ts, capture)

            attempt = success = parsed = key_matches = result_matches = source_available = pit_verified = 0
            snapshot_results: dict[str, tuple[bool, bool, set[tuple] | None]] = {}
            for digest, (_, capture) in candidates.items():
                attempt += 1
                cache = self._snapshot_cache_path(capture)
                try:
                    raw = cache.read_bytes() if cache.exists() else None
                    if raw is None:
                        r = requests.get(self._snapshot_url(capture, url), timeout=self.timeout, headers={"User-Agent": "SoccerPredictionResearch/1.0 PIT-Audit"})
                        r.raise_for_status()
                        raw = r.content
                        cache.write_bytes(raw)
                    success += 1
                    frame = pd.read_csv(BytesIO(raw))
                    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
                    if not required.issubset(frame.columns):
                        snapshot_results[digest] = (True, False, None)
                        continue
                    parsed += 1
                    keys: set[tuple] = set()
                    for r in frame.itertuples(index=False):
                        date = _date_key(getattr(r, "Date", None))
                        if date is None:
                            continue
                        try:
                            keys.add((date, str(getattr(r, "HomeTeam")).strip(), str(getattr(r, "AwayTeam")).strip(), float(getattr(r, "FTHG")), float(getattr(r, "FTAG")), str(getattr(r, "FTR")).strip()))
                        except (TypeError, ValueError):
                            continue
                    snapshot_results[digest] = (True, True, keys)
                except Exception:
                    snapshot_results[digest] = (False, False, None)

            for _, row in group.iterrows():
                key = _row_key(row)
                if key is None:
                    continue
                matched = False
                best_ts = None
                for digest, (ts, _) in candidates.items():
                    ok, parsed_ok, keys = snapshot_results.get(digest, (False, False, None))
                    if keys is not None and key in keys:
                        matched = True
                        if best_ts is None or ts < best_ts:
                            best_ts = ts
                if matched:
                    key_matches += 1
                    result_matches += 1
                    source_available += 1
                    # This adapter's evidence is deliberately tied to the archived
                    # completed result, not retrieval time. Actual prediction PIT
                    # verification is performed later against the target cutoff.
                # A row can be source-observed yet still fail a later prediction cutoff;
                # do not mark it VERIFIED here without the prediction context.
                if best_ts is not None:
                    pass

            if attempt == 0:
                stage, reason = "PIT_CUTOFF_FAILURE", "captures_exist_but_no_capture_at_or_after_event_date_end"
            elif success == 0:
                stage, reason = "SNAPSHOT_DOWNLOAD_FAILURE", "all_candidate_snapshots_failed_download_or_cache_read"
            elif parsed == 0:
                stage, reason = "SNAPSHOT_PARSE_FAILURE", "candidate_snapshots_downloaded_but_no_required_result_schema_parsed"
            elif key_matches == 0:
                stage, reason = "MATCH_KEY_MISMATCH", "parsed_snapshots_contained no matching completed-result key"
            else:
                stage, reason = "PIT_VERIFIED", "completed_result_observed_in_archived_snapshot"
                pit_verified = key_matches

            out_rows.append({"competition": competition, "season": season, "source": source, "field": "completed_result", "rows": len(group), "cdx_capture_count": cdx_count, "snapshot_attempt_count": attempt, "snapshot_success_count": success, "parse_success_count": parsed, "match_key_match_count": key_matches, "result_match_count": result_matches, "source_available_at_count": source_available, "pit_verified_count": pit_verified, "failure_stage": stage, "failure_reason": reason})

        return pd.DataFrame(out_rows, columns=columns)

def apply_pit_evidence(history: pd.DataFrame, *, cache_dir: str = "data/raw/pit_evidence", max_workers: int = DEFAULT_MAX_WORKERS) -> pd.DataFrame:
    if history.empty:
        return history.copy()
    return FootballDataWaybackAdapter(cache_dir=cache_dir, max_workers=max_workers).apply_bulk(history)

def competition_adapter_matrix() -> pd.DataFrame:
    return pd.DataFrame([{"competition": c, "source": s["source"], "adapter": s["adapter"] or "NONE", "status": "IMPLEMENTED" if s["adapter"] else "UNVERIFIED"} for c, s in COMPETITION_ADAPTERS.items()])

def build_pit_diagnostic(history: pd.DataFrame, *, cache_dir: str = "data/raw/pit_evidence", max_workers: int = DEFAULT_MAX_WORKERS) -> pd.DataFrame:
    return FootballDataWaybackAdapter(cache_dir=cache_dir, max_workers=max_workers).diagnostic_bulk(history)
