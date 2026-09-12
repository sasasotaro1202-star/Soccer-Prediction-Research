from __future__ import annotations

from src.data.pit_source_adapter_v2 import *
from src.data.pit_source_adapter_v2 import (
    FootballDataWaybackAdapter as _BaseAdapter,
    SourceEvidence,
    CaptureDiagnostic,
    _utc,
    _row_key,
    _result_lower_bound,
    COMPETITION_ADAPTERS,
)


class FootballDataWaybackAdapter(_BaseAdapter):
    """Optimized PIT replay: digest-deduplicated chronological snapshot scan."""

    def _prefetch_url(self, url, rows, workers=None):
        if not rows:
            return []
        captures = self.captures(url)
        if not captures:
            diag = self._capture_diag.get(url, CaptureDiagnostic("CDX_REQUEST_FAILURE"))
            reason = "no_archive_captures" if diag.status == "CDX_NO_CAPTURE" else f"{diag.status.lower()}: {diag.error or ''}".strip()
            return [SourceEvidence(None, "UNVERIFIABLE", reason=reason) for _ in rows]

        row_keys = [_row_key(r) for r in rows]
        bounds = [_result_lower_bound(r) for r in rows]
        unresolved = {k: i for i, k in enumerate(row_keys) if k is not None}
        results = [None] * len(rows)
        if not unresolved:
            return [SourceEvidence(None, "UNVERIFIABLE", reason="missing_record_identity") for _ in rows]

        valid_bounds = [b for b, _ in bounds if b is not None]
        if not valid_bounds:
            return [SourceEvidence(None, "UNVERIFIABLE", reason="missing_event_time") for _ in rows]
        min_bound = min(valid_bounds)

        unique = {}
        for capture in captures:
            ts = _utc(capture.get("timestamp"))
            if ts is None or ts < min_bound:
                continue
            digest = capture.get("digest") or f"{capture.get('timestamp','')}|{capture.get('original','')}"
            previous = unique.get(digest)
            if previous is None or ts < _utc(previous.get("timestamp")):
                unique[digest] = capture
        candidates = sorted(unique.values(), key=lambda c: c.get("timestamp", ""))
        if not candidates:
            return [SourceEvidence(None, "UNVERIFIABLE", reason=f"captures_exist_but_no_capture_after_result_lower_bound:{reason}") for _, reason in bounds]

        def fetch(capture):
            return capture, self._load_snapshot_keys(capture, url)

        keysets = []
        with ThreadPoolExecutor(max_workers=workers or self.max_workers) as pool:
            futures = [pool.submit(fetch, capture) for capture in candidates]
            for future in as_completed(futures):
                keysets.append(future.result())
        keysets.sort(key=lambda x: x[0].get("timestamp", ""))

        for capture, diagnostic in keysets:
            if not unresolved:
                break
            ts = _utc(capture.get("timestamp"))
            if ts is None or diagnostic.keys is None:
                continue
            for key in diagnostic.keys.intersection(unresolved.keys()):
                i = unresolved[key]
                lower_bound, bound_reason = bounds[i]
                if lower_bound is not None and ts >= lower_bound:
                    results[i] = SourceEvidence(
                        ts.isoformat(),
                        "VERIFIED",
                        self._snapshot_url(capture, url),
                        capture.get("digest"),
                        f"archived_completed_result_first_observed_after_{bound_reason.lower()}",
                    )
                    del unresolved[key]

        for i, value in enumerate(results):
            if value is not None:
                continue
            key = row_keys[i]
            lower_bound, _ = bounds[i]
            if key is None:
                reason = "missing_record_identity"
            elif lower_bound is None:
                reason = "missing_event_time"
            else:
                reason = "no_archive_snapshot_contains_completed_result"
            results[i] = SourceEvidence(None, "UNVERIFIABLE", reason=reason)
        return results

    def diagnostic_bulk(self, history):
        """Build CDX diagnostics without requiring a precomputed season_start column."""
        if history is None or history.empty:
            return pd.DataFrame(columns=["competition", "season_start", "url", "status", "capture_count", "error_type", "error"])

        work = history.copy()
        if "season_start" not in work.columns:
            if "season" not in work.columns:
                work["season_start"] = pd.NA
            else:
                work["season_start"] = pd.to_numeric(
                    work["season"].astype(str).str.extract(r"(\d{4})", expand=False),
                    errors="coerce",
                ).astype("Int64")

        rows = []
        for (competition, start_year), _group in work.groupby(["competition", "season_start"], dropna=False):
            try:
                if pd.isna(start_year):
                    raise ValueError("missing season_start")
                url = source_url(str(competition), int(start_year))
                diag = self.capture_diagnostic(url)
                rows.append({
                    "competition": competition,
                    "season_start": int(start_year),
                    "url": url,
                    "status": diag.status,
                    "capture_count": diag.capture_count,
                    "error_type": diag.error_type,
                    "error": diag.error,
                })
            except Exception as exc:
                rows.append({
                    "competition": competition,
                    "season_start": start_year,
                    "url": None,
                    "status": "ADAPTER_MAPPING_FAILURE" if not isinstance(exc, requests.RequestException) else "CDX_REQUEST_FAILURE",
                    "capture_count": 0,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
        return pd.DataFrame(rows)


def apply_pit_evidence(history, **kwargs):
    return FootballDataWaybackAdapter(**kwargs).apply_bulk(history)


def build_pit_diagnostic(history, **kwargs):
    return FootballDataWaybackAdapter(**kwargs).diagnostic_bulk(history)


def competition_adapter_matrix():
    return pd.DataFrame([{"competition": k, **v} for k, v in COMPETITION_ADAPTERS.items()])
