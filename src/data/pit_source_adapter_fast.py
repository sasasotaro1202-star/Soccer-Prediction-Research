from __future__ import annotations

import time
from io import BytesIO
import pandas as pd
import requests

from src.data.pit_source_adapter_v2 import *
from src.data.pit_source_adapter_v2 import (
    FootballDataWaybackAdapter as _BaseAdapter,
    SourceEvidence, CaptureDiagnostic, SnapshotDiagnostic,
    _utc, _row_key, _result_lower_bound, COMPETITION_ADAPTERS,
)


class FootballDataWaybackAdapter(_BaseAdapter):
    """Optimized PIT replay with resilient Wayback snapshot retrieval."""

    def __init__(self, *args, snapshot_retries: int = 4, retry_backoff: float = 1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshot_retries = max(1, int(snapshot_retries))
        self.retry_backoff = max(0.0, float(retry_backoff))

    @staticmethod
    def _with_season_start(history):
        work = history.copy()
        if "season_start" not in work.columns:
            if "season" in work.columns:
                work["season_start"] = pd.to_numeric(work["season"].astype(str).str.extract(r"(\d{4})", expand=False), errors="coerce").astype("Int64")
            else:
                work["season_start"] = pd.Series(pd.NA, index=work.index, dtype="Int64")
        else:
            work["season_start"] = pd.to_numeric(work["season_start"], errors="coerce").astype("Int64")
        return work

    @staticmethod
    def _search_floor(row, conservative_bound):
        kickoff = _utc(row.get("kickoff_utc"))
        if kickoff is None or not bool(row.get("kickoff_time_available", False)):
            return conservative_bound
        return kickoff

    def _load_snapshot_keys(self, capture, original_url):
        """Retry transient Wayback snapshot failures without changing PIT semantics."""
        identity = f"{capture.get('digest','')}|{capture.get('timestamp','')}|{capture.get('original','')}"
        if identity in self._snapshot_diag:
            return self._snapshot_diag[identity]
        cache = self._snapshot_cache_path(capture)
        raw = None
        last_error = None
        for attempt in range(1, self.snapshot_retries + 1):
            try:
                if cache.exists():
                    raw = cache.read_bytes()
                else:
                    response = requests.get(self._snapshot_url(capture, original_url), timeout=self.timeout, headers={"User-Agent": "SoccerPredictionResearch/1.0 PIT-Audit"})
                    response.raise_for_status()
                    raw = response.content
                    cache.write_bytes(raw)
                break
            except (requests.RequestException, OSError) as exc:
                last_error = exc
                if isinstance(exc, OSError) and cache.exists() and raw is None:
                    try: cache.unlink()
                    except OSError: pass
                if attempt < self.snapshot_retries:
                    time.sleep(self.retry_backoff * attempt)
        if raw is None:
            diag = SnapshotDiagnostic("SNAPSHOT_DOWNLOAD_FAILURE", error_type=type(last_error).__name__ if last_error else "UnknownError", error=f"after_{self.snapshot_retries}_attempts: {last_error}")
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
            diag = SnapshotDiagnostic("SNAPSHOT_SCHEMA_FAILURE", error_type="MissingColumns", error=",".join(sorted(required - set(frame.columns))))
            self._snapshot_diag[identity] = diag
            return diag
        keys = set()
        for r in frame.itertuples(index=False):
            date = self._date_key(getattr(r, "Date", None))
            if date is None: continue
            try:
                keys.add((date, str(getattr(r, "HomeTeam")).strip(), str(getattr(r, "AwayTeam")).strip(), float(getattr(r, "FTHG")), float(getattr(r, "FTAG")), str(getattr(r, "FTR")).strip()))
            except (TypeError, ValueError): continue
        diag = SnapshotDiagnostic("SNAPSHOT_PARSED", keys=keys)
        self._snapshot_diag[identity] = diag
        return diag

    def _prefetch_url(self, url, rows, workers=None):
        if not rows: return []
        captures = self.captures(url)
        if not captures:
            diag = self._capture_diag.get(url, CaptureDiagnostic("CDX_REQUEST_FAILURE"))
            reason = "no_archive_captures" if diag.status == "CDX_NO_CAPTURE" else f"{diag.status.lower()}: {diag.error or ''}".strip()
            return [SourceEvidence(None, "UNVERIFIABLE", reason=reason) for _ in rows]
        row_keys = [_row_key(r) for r in rows]
        bounds = [_result_lower_bound(r) for r in rows]
        results = [None] * len(rows)
        unresolved = {k: i for i, k in enumerate(row_keys) if k is not None}
        valid_bounds = [b for b, _ in bounds if b is not None]
        if not unresolved: return [SourceEvidence(None, "UNVERIFIABLE", reason="missing_record_identity") for _ in rows]
        if not valid_bounds: return [SourceEvidence(None, "UNVERIFIABLE", reason="missing_event_time") for _ in rows]
        search_bounds = [self._search_floor(r, b) for r, (b, _) in zip(rows, bounds)]
        valid_search_bounds = [b for b in search_bounds if b is not None]
        min_search_bound = min(valid_search_bounds) if valid_search_bounds else min(valid_bounds)
        unique = {}
        for capture in captures:
            ts = _utc(capture.get("timestamp"))
            if ts is None or ts < min_search_bound: continue
            digest = capture.get("digest") or f"{capture.get('timestamp','')}|{capture.get('original','')}"
            previous = unique.get(digest)
            if previous is None or (_utc(previous.get("timestamp")) or ts) > ts: unique[digest] = capture
        candidates = sorted(unique.values(), key=lambda c: c.get("timestamp", ""))
        if not candidates:
            reasons = ";".join(sorted(set(reason for _, reason in bounds)))
            return [SourceEvidence(None, "UNVERIFIABLE", reason=f"captures_exist_but_no_capture_after_result_lower_bound:{reasons}") for _ in rows]
        def fetch(capture): return capture, self._load_snapshot_keys(capture, url)
        keysets = []
        with ThreadPoolExecutor(max_workers=workers or self.max_workers) as pool:
            futures = [pool.submit(fetch, capture) for capture in candidates]
            for future in as_completed(futures): keysets.append(future.result())
        keysets.sort(key=lambda x: x[0].get("timestamp", ""))
        def scan(allow_early_precise_capture=False):
            for capture, diagnostic in keysets:
                if not unresolved: break
                ts = _utc(capture.get("timestamp"))
                if ts is None or diagnostic.keys is None: continue
                for key in diagnostic.keys.intersection(unresolved.keys()):
                    i = unresolved[key]
                    conservative_bound, bound_reason = bounds[i]
                    if conservative_bound is None: continue
                    accepted_bound = conservative_bound
                    accepted_reason = bound_reason.lower()
                    if allow_early_precise_capture and bool(rows[i].get("kickoff_time_available", False)):
                        kickoff = _utc(rows[i].get("kickoff_utc"))
                        if kickoff is not None and kickoff <= ts < conservative_bound:
                            accepted_bound = kickoff; accepted_reason = "kickoff"
                    if ts >= accepted_bound:
                        results[i] = SourceEvidence(ts.isoformat(), "VERIFIED", self._snapshot_url(capture, url), capture.get("digest"), f"archived_completed_result_first_observed_after_{accepted_reason}")
                        del unresolved[key]
        scan(False)
        if unresolved: scan(True)
        snapshot_errors = sorted({d.status for _, d in keysets if d.keys is None and d.status})
        for i, value in enumerate(results):
            if value is not None: continue
            key = row_keys[i]; lower_bound, _ = bounds[i]
            if key is None: reason = "missing_record_identity"
            elif lower_bound is None: reason = "missing_event_time"
            elif snapshot_errors: reason = "no_archive_snapshot_contains_completed_result:" + ",".join(snapshot_errors)
            else: reason = "no_archive_snapshot_contains_completed_result"
            results[i] = SourceEvidence(None, "UNVERIFIABLE", reason=reason)
        return results

    def apply_bulk(self, history):
        if history is None or history.empty: return history.copy() if history is not None else history
        return super().apply_bulk(self._with_season_start(history))

    def diagnostic_bulk(self, history):
        if history is None or history.empty:
            return pd.DataFrame(columns=["competition", "season_start", "url", "status", "capture_count", "error_type", "error", "cdx_status", "failure_stage", "failure_reason"])
        work = self._with_season_start(history); rows = []
        for (competition, start_year), _group in work.groupby(["competition", "season_start"], dropna=False):
            try:
                if pd.isna(start_year): raise ValueError("missing season_start")
                url = source_url(str(competition), int(start_year)); diag = self.capture_diagnostic(url); cdx_status = diag.status
                rows.append({"competition": competition, "season_start": int(start_year), "url": url, "status": diag.status, "capture_count": diag.capture_count, "error_type": diag.error_type, "error": diag.error, "cdx_status": cdx_status, "failure_stage": cdx_status if cdx_status != "CDX_CAPTURE_FOUND" else "", "failure_reason": diag.error or ("no archive capture" if cdx_status == "CDX_NO_CAPTURE" else "")})
            except Exception as exc:
                reason = str(exc)
                rows.append({"competition": competition, "season_start": start_year, "url": None, "status": "ADAPTER_MAPPING_FAILURE", "capture_count": 0, "error_type": type(exc).__name__, "error": reason, "cdx_status": "NOT_ATTEMPTED", "failure_stage": "ADAPTER_MAPPING_FAILURE", "failure_reason": reason})
        return pd.DataFrame(rows)


def apply_pit_evidence(history, **kwargs): return FootballDataWaybackAdapter(**kwargs).apply_bulk(history)
def build_pit_diagnostic(history, **kwargs): return FootballDataWaybackAdapter(**kwargs).diagnostic_bulk(history)
def competition_adapter_matrix(): return pd.DataFrame([{"competition": k, **v} for k, v in COMPETITION_ADAPTERS.items()])
