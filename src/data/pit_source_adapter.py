"""Compatibility exports for the optimized, time-precise PIT archive adapter."""

import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.data import pit_source_adapter_v2 as _impl
from src.data.pit_source_adapter_fast import *
from src.data.pit_source_adapter_fast import FootballDataWaybackAdapter as _FastFootballDataWaybackAdapter
from src.data.pit_source_adapter_v2 import COMPETITION_ADAPTERS, _result_lower_bound


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


def _date_key(value: Any) -> str | None:
    """Parse source dates without interpreting ISO YYYY-MM-DD as day-first."""
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
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
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


_impl._utc = _utc
_impl._date_key = _date_key


class FootballDataWaybackAdapter(_FastFootballDataWaybackAdapter):
    """Stable PIT adapter with low-concurrency retrieval and CDX retry hardening."""

    def __init__(self, *args, cdx_retries: int = 4, cdx_retry_backoff: float = 1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_workers = min(self.max_workers, 2)
        self.cdx_retries = max(1, int(cdx_retries))
        self.cdx_retry_backoff = max(0.0, float(cdx_retry_backoff))

    def captures(self, url: str):
        """Retry transient CDX failures; never turn a transport failure into no-capture."""
        last = None
        for attempt in range(1, self.cdx_retries + 1):
            self._captures.pop(url, None)
            rows = super().captures(url)
            diag = self._capture_diag.get(url)
            last = diag
            if rows or diag is None or diag.status != "CDX_REQUEST_FAILURE":
                return rows
            if attempt < self.cdx_retries:
                time.sleep(self.cdx_retry_backoff * attempt)
        return self._captures.get(url, [])

    def diagnostic_bulk(self, history: pd.DataFrame) -> pd.DataFrame:
        rows = []
        if history.empty:
            return pd.DataFrame(columns=[
                "competition", "season_start", "url", "status", "capture_count",
                "error_type", "error", "cdx_status", "failure_stage", "failure_reason",
            ])

        work = history.copy()
        if "season_start" not in work.columns and "season" in work.columns:
            work["season_start"] = work["season"].astype(str).str.extract(r"(\d{4})", expand=False)
        if "season_start" in work.columns:
            work["season_start"] = pd.to_numeric(work["season_start"], errors="coerce")

        for (competition, start_year), group in work.groupby(
            ["competition", "season_start"], dropna=False
        ):
            try:
                url = source_url(str(competition), int(start_year))
                diag = self.capture_diagnostic(url)
                cdx_status = diag.status
                failure_stage = cdx_status if cdx_status != "CDX_CAPTURE_FOUND" else ""
                failure_reason = diag.error or ("no archive capture" if cdx_status == "CDX_NO_CAPTURE" else "")
                rows.append({
                    "competition": competition,
                    "season_start": start_year,
                    "url": url,
                    "status": diag.status,
                    "capture_count": diag.capture_count,
                    "error_type": diag.error_type,
                    "error": diag.error,
                    "cdx_status": cdx_status,
                    "failure_stage": failure_stage,
                    "failure_reason": failure_reason,
                })
            except Exception as exc:
                reason = str(exc)
                rows.append({
                    "competition": competition,
                    "season_start": start_year,
                    "url": None,
                    "status": "ADAPTER_MAPPING_FAILURE",
                    "capture_count": 0,
                    "error_type": type(exc).__name__,
                    "error": reason,
                    "cdx_status": "NOT_ATTEMPTED",
                    "failure_stage": "ADAPTER_MAPPING_FAILURE",
                    "failure_reason": reason,
                })
        return pd.DataFrame(rows)


def competition_adapter_matrix() -> pd.DataFrame:
    rows = []
    for competition, spec in COMPETITION_ADAPTERS.items():
        rows.append({
            "competition": competition,
            "source": spec.get("source"),
            "source_code": spec.get("source_code"),
            "adapter": spec.get("adapter"),
            "status": "IMPLEMENTED" if spec.get("adapter") else "UNVERIFIED",
        })
    return pd.DataFrame(rows)


def build_pit_diagnostic(history: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return FootballDataWaybackAdapter(**kwargs).diagnostic_bulk(history)
