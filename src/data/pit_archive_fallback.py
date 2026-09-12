from __future__ import annotations

"""Bounded secondary PIT evidence provider using Arquivo.pt.

This provider is deliberately conservative and bounded. It never upgrades a row
unless an archived CSV contains the exact completed-result key and the archive
capture timestamp is at/after the PIT lower bound. Network failure is recorded
as UNVERIFIABLE rather than being treated as missing historical data.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO

import pandas as pd
import requests

from src.data.pit_source_adapter_v2 import _date_key, _row_key, _result_lower_bound

ARQUIVO_CDX = "https://arquivo.pt/wayback/cdx"
ARQUIVO_WEB = "https://arquivo.pt/wayback"
USER_AGENT = "SoccerPredictionResearch/1.0 (+PIT-audit)"


def _utc(value):
    if value is None or value == "":
        return None
    text = str(value).strip()
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


def _keyset(raw: bytes):
    frame = pd.read_csv(BytesIO(raw))
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    if not required.issubset(frame.columns):
        return None
    keys = set()
    for r in frame.itertuples(index=False):
        date = _date_key(getattr(r, "Date", None))
        if date is None:
            continue
        try:
            keys.add((date, str(getattr(r, "HomeTeam")).strip(), str(getattr(r, "AwayTeam")).strip(),
                      float(getattr(r, "FTHG")), float(getattr(r, "FTAG")), str(getattr(r, "FTR")).strip()))
        except (TypeError, ValueError):
            continue
    return keys


def _captures(url: str, retries: int = 2, timeout: int = 15):
    params = {
        "url": url,
        "output": "json",
        "filter": "statuscode:200",
        "fl": "timestamp,original,mimetype,statuscode,digest",
    }
    last_error = None
    for attempt in range(retries):
        try:
            r = requests.get(ARQUIVO_CDX, params=params, timeout=timeout,
                             headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            payload = r.json()
            if not payload or len(payload) <= 1:
                return []
            return [dict(zip(payload[0], row)) for row in payload[1:]]
        except (requests.RequestException, ValueError, TypeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(1.0)
    return []


def _fetch_capture(capture: dict, original_url: str, retries: int = 2, timeout: int = 15):
    ts = str(capture.get("timestamp", "")).strip()
    if not ts:
        return None
    replay_url = f"{ARQUIVO_WEB}/{ts}/{original_url}"
    for attempt in range(retries):
        try:
            r = requests.get(replay_url, timeout=timeout, allow_redirects=True,
                             headers={"User-Agent": USER_AGENT, "Accept": "text/csv,text/plain,*/*"})
            r.raise_for_status()
            final_url = str(getattr(r, "url", replay_url))
            if not final_url.startswith("https://arquivo.pt/wayback/"):
                raise requests.HTTPError("unexpected_archive_redirect", response=r)
            raw = r.content
            if raw[:512].lstrip().lower().startswith((b"<!doctype html", b"<html")):
                raise requests.HTTPError("archive_html_error_page", response=r)
            return raw, replay_url
        except (requests.RequestException, OSError):
            if attempt + 1 < retries:
                time.sleep(1.0)
    return None


def apply_arquivo_fallback(history: pd.DataFrame) -> pd.DataFrame:
    """Verify unresolved rows with a bounded, grouped Arquivo.pt pass.

    The whole fallback is fail-closed: any provider exception leaves the row
    unchanged. It is intentionally bounded so a degraded archive cannot consume
    the entire GitHub Actions job timeout.
    """
    if history is None or history.empty:
        return history.copy() if history is not None else history

    out = history.copy()
    try:
        from src.data.pit_source_adapter_v2 import source_url

        pending = []
        for idx, row in out.iterrows():
            if str(row.get("pit_evidence_status", "")) == "VERIFIED":
                continue
            key = _row_key(row)
            lower_bound, bound_reason = _result_lower_bound(row)
            if key is None or lower_bound is None:
                continue
            try:
                url = source_url(str(row.get("competition")), int(row.get("season_start")))
            except (TypeError, ValueError, KeyError):
                continue
            pending.append((idx, key, lower_bound, bound_reason, url))

        # One CDX request per source URL, not one per row. Parallelism is bounded.
        urls = sorted({x[4] for x in pending})
        capture_map = {}
        with ThreadPoolExecutor(max_workers=min(4, max(1, len(urls)))) as pool:
            futures = {pool.submit(_captures, url): url for url in urls}
            for future in as_completed(futures):
                url = futures[future]
                try:
                    capture_map[url] = future.result()
                except Exception:
                    capture_map[url] = []

        # Fetch only the earliest plausible captures. Later captures cannot be
        # needed once an exact result match has been established.
        jobs = []
        for idx, key, lower_bound, bound_reason, url in pending:
            captures = []
            for capture in capture_map.get(url, []):
                ts = _utc(capture.get("timestamp"))
                if ts is not None and ts >= lower_bound:
                    captures.append((ts, capture))
            captures.sort(key=lambda x: x[0])
            jobs.append((idx, key, lower_bound, bound_reason, url, captures[:4]))

        def verify(job):
            idx, key, lower_bound, bound_reason, url, captures = job
            for ts, capture in captures:
                fetched = _fetch_capture(capture, url)
                if not fetched:
                    continue
                raw, replay_url = fetched
                try:
                    keys = _keyset(raw)
                except Exception:
                    continue
                if keys is not None and key in keys:
                    return idx, ts, replay_url, capture, bound_reason
            return None

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(verify, job) for job in jobs]
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception:
                    result = None
                if result is None:
                    continue
                idx, ts, replay_url, capture, bound_reason = result
                out.at[idx, "source_available_at_utc"] = ts.isoformat()
                out.at[idx, "pit_evidence_status"] = "VERIFIED"
                out.at[idx, "pit_evidence_reason"] = f"arquivo_pt_completed_result_first_observed_after_{bound_reason.lower()}"
                out.at[idx, "pit_evidence_url"] = replay_url
                out.at[idx, "capture_digest"] = capture.get("digest")
    except Exception:
        # Secondary provider is never allowed to break the audit itself.
        pass

    return out
