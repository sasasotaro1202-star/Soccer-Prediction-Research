from __future__ import annotations

import os
import re
import time
import unicodedata
from io import BytesIO
import pandas as pd
import requests

from src.data.http_resilience import resilient_get

from src.data.pit_source_adapter_v2 import *
from src.data.pit_source_adapter_v2 import (
    FootballDataWaybackAdapter as _BaseAdapter,
    SourceEvidence, CaptureDiagnostic, SnapshotDiagnostic,
    _utc, _row_key, _result_lower_bound, _date_key, COMPETITION_ADAPTERS,
)

_DATE_KEY = _date_key

# Persist only immutable VERIFIED evidence so interrupted archive replay can resume
# without trusting prior UNVERIFIABLE results. Bump when verification semantics change.
PIT_EVIDENCE_CACHE_VERSION = 1


def normalize_team_identity(value: object) -> str:
    """Normalize presentation-level team-name differences only."""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9]+", "", text).casefold()


def _normalized_row_key(row: pd.Series) -> tuple | None:
    date = _date_key(row.get("source_event_date")) or _date_key(row.get("kickoff_utc"))
    try:
        return (date, normalize_team_identity(row.get("home_team", "")), normalize_team_identity(row.get("away_team", "")), float(row.get("home_goals")), float(row.get("away_goals")), str(row.get("result", "")).strip()) if date else None
    except (TypeError, ValueError):
        return None


class FootballDataWaybackAdapter(_BaseAdapter):
    """Optimized PIT replay with resilient retrieval and earliest-capture semantics."""

    _date_key = staticmethod(_DATE_KEY)

    def __init__(self, *args, snapshot_retries: int = 6, retry_backoff: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshot_retries = max(1, int(snapshot_retries))
        self.retry_backoff = max(0.0, float(retry_backoff))

    @staticmethod
    def _snapshot_url(capture, original_url):
        timestamp = str(capture.get("timestamp", "")).strip()
        target = str(original_url or "").strip()
        if not timestamp or not target:
            raise ValueError("Wayback capture URL requires timestamp and original URL")
        return f"{WAYBACK_WEB}/{timestamp}id_/{target}"

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
        identity = f"{capture.get('digest','')}|{capture.get('timestamp','')}|{capture.get('original','')}"
        if identity in self._snapshot_diag:
            return self._snapshot_diag[identity]

        cache = self._snapshot_cache_path(capture)
        capture_original = str(capture.get("original", "")).strip()
        snapshot_urls = []
        if capture_original:
            snapshot_urls.append(self._snapshot_url(capture, capture_original))
        if original_url and original_url != capture_original:
            snapshot_urls.append(self._snapshot_url(capture, original_url))
        if not snapshot_urls:
            raise ValueError("Wayback replay requires an original URL")

        last_error = None
        last_failure_status = "SNAPSHOT_DOWNLOAD_FAILURE"

        for attempt in range(1, self.snapshot_retries + 1):
            raw = None
            loaded_from_cache = False
            try:
                if cache.exists():
                    raw = cache.read_bytes()
                    loaded_from_cache = True
                else:
                    for snapshot_url in snapshot_urls:
                        try:
                            response = resilient_get(
                                requests.get,
                                snapshot_url,
                                timeout=self.timeout,
                                retries=1,
                                backoff=self.retry_backoff,
                                headers={"User-Agent": "SoccerPredictionResearch/1.0 PIT-Audit"},
                            )
                            candidate = response.content
                            if candidate[:512].lstrip().lower().startswith((b"<!doctype html", b"<html")):
                                raise ValueError("html_instead_of_snapshot")
                            raw = candidate
                            # Persist only content that is at least non-HTML; parsing
                            # below is still authoritative before a snapshot is accepted.
                            cache.write_bytes(raw)
                            last_error = None
                            break
                        except (requests.RequestException, OSError, ValueError) as exc:
                            last_error = exc
                    if raw is None:
                        if last_error is None:
                            raise RuntimeError("snapshot_download_returned_no_content")
                        raise last_error

                if raw[:512].lstrip().lower().startswith((b"<!doctype html", b"<html")):
                    last_failure_status = "SNAPSHOT_PARSE_FAILURE"
                    raise ValueError("html_instead_of_snapshot")

                frame = pd.read_csv(BytesIO(raw))
                required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
                if not required.issubset(frame.columns):
                    last_failure_status = "SNAPSHOT_SCHEMA_FAILURE"
                    raise ValueError(
                        "MissingColumns:" + ",".join(sorted(required - set(frame.columns)))
                    )

                keys = set()
                for r in frame.itertuples(index=False):
                    date = self._date_key(getattr(r, "Date", None))
                    if date is None:
                        continue
                    try:
                        keys.add((
                            date,
                            normalize_team_identity(getattr(r, "HomeTeam")),
                            normalize_team_identity(getattr(r, "AwayTeam")),
                            float(getattr(r, "FTHG")),
                            float(getattr(r, "FTAG")),
                            str(getattr(r, "FTR")).strip(),
                        ))
                    except (TypeError, ValueError):
                        continue

                diag = SnapshotDiagnostic("SNAPSHOT_PARSED", keys=keys)
                self._snapshot_diag[identity] = diag
                return diag

            except (requests.RequestException, OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                last_error = exc
                # A malformed cached response can otherwise make every future retry
                # deterministically fail. Remove only the specific capture cache so
                # the next bounded attempt can re-fetch immutable evidence.
                if loaded_from_cache and cache.exists():
                    try:
                        cache.unlink()
                    except OSError:
                        pass
                if attempt < self.snapshot_retries:
                    time.sleep(self.retry_backoff * attempt)

        if last_failure_status == "SNAPSHOT_SCHEMA_FAILURE":
            diag = SnapshotDiagnostic(
                "SNAPSHOT_SCHEMA_FAILURE",
                error_type=type(last_error).__name__ if last_error else "MissingColumns",
                error=str(last_error) if last_error else "snapshot schema invalid",
            )
        elif last_failure_status == "SNAPSHOT_PARSE_FAILURE":
            diag = SnapshotDiagnostic(
                "SNAPSHOT_PARSE_FAILURE",
                error_type=type(last_error).__name__ if last_error else "ValueError",
                error=str(last_error) if last_error else "snapshot parse failed",
            )
        else:
            diag = SnapshotDiagnostic(
                "SNAPSHOT_DOWNLOAD_FAILURE",
                error_type=type(last_error).__name__ if last_error else "UnknownError",
                error=f"after_{self.snapshot_retries}_attempts: {last_error}",
            )
        self._snapshot_diag[identity] = diag
        return diag

    def _evidence_cache_path(self, url, row, lower_bound):
        """Return a deterministic row-level cache path for immutable VERIFIED evidence."""
        key = {
            "version": PIT_EVIDENCE_CACHE_VERSION,
            "url": str(url),
            "row_key": _normalized_row_key(row),
            "match_id": row.get("match_id", ""),
            "source_name": row.get("source_name", ""),
            "source_record_id": row.get("source_record_id", ""),
            "lower_bound": lower_bound.isoformat() if lower_bound is not None else None,
        }
        digest = self._cache_key(json.dumps(key, sort_keys=True, default=str))
        return self.cache_dir / f"evidence_{digest}.json"

    def _load_verified_evidence_cache(self, url, row, lower_bound):
        """Load a previously VERIFIED result only after rechecking its PIT boundary."""
        if lower_bound is None:
            return None
        path = self._evidence_cache_path(url, row, lower_bound)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            return None
        if payload.get("cache_version") != PIT_EVIDENCE_CACHE_VERSION:
            return None
        if payload.get("status") != "VERIFIED":
            return None
        timestamp = _utc(payload.get("source_available_at_utc"))
        if timestamp is None or timestamp < lower_bound:
            return None
        expected_key = _normalized_row_key(row)
        if payload.get("row_key") != list(expected_key or ()):
            return None
        if payload.get("url") != str(url):
            return None
        for field in ("match_id", "source_name", "source_record_id"):
            if str(payload.get(field, "")) != str(row.get(field, "")):
                return None
        return SourceEvidence(
            timestamp.isoformat(),
            "VERIFIED",
            payload.get("evidence_url"),
            payload.get("capture_digest"),
            payload.get("reason", "verified_evidence_cache"),
        )

    def _save_verified_evidence_cache(self, url, row, lower_bound, evidence):
        """Persist only a PIT-revalidated VERIFIED evidence record; failures stay transient."""
        if evidence is None or evidence.evidence_status != "VERIFIED":
            return
        timestamp = _utc(evidence.source_available_at_utc)
        row_key = _normalized_row_key(row)
        if lower_bound is None or timestamp is None or row_key is None or timestamp < lower_bound:
            return
        path = self._evidence_cache_path(url, row, lower_bound)
        payload = {
            "cache_version": PIT_EVIDENCE_CACHE_VERSION,
            "status": "VERIFIED",
            "url": str(url),
            "row_key": list(row_key),
            "match_id": row.get("match_id", ""),
            "source_name": row.get("source_name", ""),
            "source_record_id": row.get("source_record_id", ""),
            "lower_bound": lower_bound.isoformat(),
            "source_available_at_utc": timestamp.isoformat(),
            "evidence_url": evidence.evidence_url,
            "capture_digest": evidence.capture_digest,
            "reason": evidence.reason,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            try:
                tmp.unlink()
            except OSError:
                pass

    def _prefetch_url(self, url, rows, workers=None):
        if not rows:
            return []

        row_keys = [_normalized_row_key(r) for r in rows]
        bounds = [_result_lower_bound(r) for r in rows]
        results = [None] * len(rows)
        unresolved = {i for i, k in enumerate(row_keys) if k is not None}

        # Resume from immutable VERIFIED evidence first. Cached failures are never
        # consulted, so a later archive update can still turn an earlier miss into PASS.
        for i, (_, lower_bound) in enumerate(bounds):
            if i not in unresolved or lower_bound is None:
                continue
            cached = self._load_verified_evidence_cache(url, rows[i], lower_bound)
            if cached is not None:
                results[i] = cached
                unresolved.remove(i)

        # A fully cached batch must resume without touching the archive control plane.
        if not unresolved:
            return results

        valid_bounds = [bounds[i][0] for i in unresolved if bounds[i][0] is not None]
        if not valid_bounds:
            return [
                results[i]
                if results[i] is not None
                else SourceEvidence(None, "UNVERIFIABLE", reason="missing_event_time")
                for i in range(len(rows))
            ]

        captures = self.captures(url)
        if not captures:
            diag = self._capture_diag.get(url, CaptureDiagnostic("CDX_REQUEST_FAILURE"))
            reason = (
                "no_archive_captures"
                if diag.status == "CDX_NO_CAPTURE"
                else f"{diag.status.lower()}: {diag.error or ''}".strip()
            )
            return [
                results[i]
                if results[i] is not None
                else SourceEvidence(None, "UNVERIFIABLE", reason=reason)
                for i in range(len(rows))
            ]

        search_bounds = [
            self._search_floor(rows[i], bounds[i][0])
            for i in unresolved
        ]
        valid_search_bounds = [b for b in search_bounds if b is not None]
        min_search_bound = (
            min(valid_search_bounds)
            if valid_search_bounds
            else min(valid_bounds)
        )

        unique = {}
        for capture in captures:
            ts = _utc(capture.get("timestamp"))
            if ts is None or ts < min_search_bound:
                continue
            digest = capture.get("digest") or f"{capture.get('timestamp','')}|{capture.get('original','')}"
            previous = unique.get(digest)
            # PIT requires the earliest observation of identical content, not the latest.
            if previous is None or (_utc(previous.get("timestamp")) or ts) > ts:
                unique[digest] = capture
        candidates = sorted(unique.values(), key=lambda c: c.get("timestamp", ""))
        if not candidates:
            reasons = ";".join(
                sorted({bounds[i][1] for i in unresolved if bounds[i][1]})
            )
            return [
                results[i]
                if results[i] is not None
                else SourceEvidence(
                    None,
                    "UNVERIFIABLE",
                    reason=f"captures_exist_but_no_capture_after_result_lower_bound:{reasons}",
                )
                for i in range(len(rows))
            ]

        # Fetch captures in bounded chronological batches. The previous implementation
        # submitted every candidate snapshot at once, even when an early capture had
        # already resolved all pending rows. Batch ordering preserves the earliest-
        # observation PIT semantics while bounding unnecessary network/parse work.
        try:
            batch_size = int(os.getenv("PIT_CAPTURE_BATCH_SIZE", "32"))
        except ValueError:
            batch_size = 32
        batch_size = max(1, min(64, batch_size))

        snapshot_errors = set()

        def fetch(capture):
            return capture, self._load_snapshot_keys(capture, url)

        for offset in range(0, len(candidates), batch_size):
            if not unresolved:
                break

            batch = candidates[offset : offset + batch_size]
            with ThreadPoolExecutor(max_workers=workers or self.max_workers) as pool:
                futures = [pool.submit(fetch, capture) for capture in batch]
                keysets = []
                for future in futures:
                    try:
                        keysets.append(future.result())
                    except Exception as exc:
                        snapshot_errors.add(
                            f"SNAPSHOT_FETCH_TASK_FAILURE:{type(exc).__name__}"
                        )

            keysets.sort(key=lambda x: x[0].get("timestamp", ""))

            # Scan only the completed batch in chronological order. Later batches
            # are not downloaded once every unresolved row has been verified.
            for capture, diagnostic in keysets:
                if not unresolved:
                    break
                ts = _utc(capture.get("timestamp"))
                if diagnostic.keys is None:
                    if getattr(diagnostic, "status", None):
                        snapshot_errors.add(str(diagnostic.status))
                    continue
                matching_indices = [i for i in unresolved if row_keys[i] in diagnostic.keys]
                for i in matching_indices:
                    conservative_bound, bound_reason = bounds[i]
                    if conservative_bound is None:
                        continue
                    accepted_bound = conservative_bound
                    accepted_reason = bound_reason.lower()
                    if ts is not None and ts >= accepted_bound:
                        results[i] = SourceEvidence(
                            ts.isoformat(),
                            "VERIFIED",
                            self._snapshot_url(capture, url),
                            capture.get("digest"),
                            f"archived_completed_result_first_observed_after_{accepted_reason}",
                        )
                        self._save_verified_evidence_cache(url, rows[i], conservative_bound, results[i])
                        unresolved.remove(i)

        # PIT is fail-closed: a capture before the conservative publication lower
        # bound cannot be accepted merely because it already contains the final score.
        for i, value in enumerate(results):
            if value is not None:
                continue
            key = row_keys[i]
            lower_bound, _ = bounds[i]
            if key is None:
                reason = "missing_record_identity"
            elif lower_bound is None:
                reason = "missing_event_time"
            elif snapshot_errors:
                reason = "no_archive_snapshot_contains_completed_result:" + ",".join(sorted(snapshot_errors))
            else:
                reason = "no_archive_snapshot_contains_completed_result"
            results[i] = SourceEvidence(None, "UNVERIFIABLE", reason=reason)
        return results

    def apply_bulk(self, history):
        if history is None or history.empty:
            return history.copy() if history is not None else history
        return super().apply_bulk(self._with_season_start(history))

    def diagnostic_bulk(self, history):
        if history is None or history.empty:
            return pd.DataFrame(columns=["competition", "season_start", "url", "status", "capture_count", "error_type", "error", "cdx_status", "failure_stage", "failure_reason"])
        work = self._with_season_start(history)
        rows = []
        for (competition, start_year), _group in work.groupby(["competition", "season_start"], dropna=False):
            try:
                if pd.isna(start_year):
                    raise ValueError("missing season_start")
                url = source_url(str(competition), int(start_year))
                diag = self.capture_diagnostic(url)
                cdx_status = diag.status
                rows.append({"competition": competition, "season_start": int(start_year), "url": url, "status": diag.status, "capture_count": diag.capture_count, "error_type": diag.error_type, "error": diag.error, "cdx_status": cdx_status, "failure_stage": cdx_status if cdx_status != "CDX_CAPTURE_FOUND" else "", "failure_reason": diag.error or ("no archive capture" if cdx_status == "CDX_NO_CAPTURE" else "")})
            except Exception as exc:
                reason = str(exc)
                rows.append({"competition": competition, "season_start": start_year, "url": None, "status": "ADAPTER_MAPPING_FAILURE", "capture_count": 0, "error_type": type(exc).__name__, "error": reason, "cdx_status": "NOT_ATTEMPTED", "failure_stage": "ADAPTER_MAPPING_FAILURE", "failure_reason": reason})
        return pd.DataFrame(rows)


def apply_pit_evidence(history, **kwargs):
    return FootballDataWaybackAdapter(**kwargs).apply_bulk(history)


def build_pit_diagnostic(history, **kwargs):
    return FootballDataWaybackAdapter(**kwargs).diagnostic_bulk(history)


def competition_adapter_matrix():
    return pd.DataFrame([{"competition": k, **v} for k, v in COMPETITION_ADAPTERS.items()])
