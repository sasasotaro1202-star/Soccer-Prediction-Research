from __future__ import annotations

"""Secondary PIT evidence provider using Arquivo.pt.

This module is deliberately conservative: it is only used after the primary
Internet Archive replay cannot verify a row. A row is VERIFIED only when an
Arquivo.pt CDX capture is found, the archived CSV is successfully retrieved
and parsed, the exact completed-result key is present, and the archive
capture timestamp is at/after the PIT lower bound.
"""

import time
from io import BytesIO
from datetime import datetime, timezone
from urllib.parse import quote

import pandas as pd
import requests

from src.data.pit_source_adapter_v2 import _date_key, _row_key, _result_lower_bound, SourceEvidence

ARQUIVO_CDX = "https://arquivo.pt/wayback/cdx"
ARQUIVO_WEB = "https://arquivo.pt/wayback"


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


def _captures(url: str, session: requests.Session, retries: int = 5):
    params = {
        "url": url,
        "output": "json",
        "filter": "statuscode:200",
        "fl": "timestamp,original,mimetype,statuscode,digest",
    }
    last = None
    for attempt in range(1, retries + 1):
        try:
            r = session.get(ARQUIVO_CDX, params=params, timeout=45,
                            headers={"User-Agent": "SoccerPredictionResearch/1.0 (+PIT-audit)"})
            r.raise_for_status()
            payload = r.json()
            if not payload or len(payload) <= 1:
                return []
            return [dict(zip(payload[0], row)) for row in payload[1:]]
        except (requests.RequestException, ValueError, TypeError) as exc:
            last = exc
            if attempt < retries:
                time.sleep(min(20.0, 1.5 * attempt))
    return []


def _fetch_capture(capture: dict, original_url: str, session: requests.Session, retries: int = 5):
    ts = str(capture.get("timestamp", "")).strip()
    if not ts:
        return None
    replay_url = f"{ARQUIVO_WEB}/{ts}/{original_url}"
    last = None
    for attempt in range(1, retries + 1):
        try:
            r = session.get(replay_url, timeout=45, allow_redirects=True,
                            headers={"User-Agent": "SoccerPredictionResearch/1.0 (+PIT-audit)",
                                     "Accept": "text/csv,text/plain,*/*"})
            r.raise_for_status()
            final_url = str(getattr(r, "url", replay_url))
            if not final_url.startswith("https://arquivo.pt/wayback/"):
                raise requests.HTTPError("unexpected_archive_redirect", response=r)
            raw = r.content
            if raw[:512].lstrip().lower().startswith(b"<!doctype html"):
                raise requests.HTTPError("archive_html_error_page", response=r)
            return raw, replay_url
        except (requests.RequestException, OSError) as exc:
            last = exc
            if attempt < retries:
                time.sleep(min(20.0, 1.5 * attempt))
    return None


def apply_arquivo_fallback(history: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty:
        return history.copy() if history is not None else history
    out = history.copy()
    session = requests.Session()
    for idx, row in out.iterrows():
        if str(row.get("pit_evidence_status", "")) == "VERIFIED":
            continue
        key = _row_key(row)
        lower_bound, bound_reason = _result_lower_bound(row)
        if key is None or lower_bound is None:
            continue
        try:
            from src.data.pit_source_adapter_v2 import source_url
            url = source_url(str(row.get("competition")), int(row.get("season_start")))
        except Exception:
            continue
        captures = _captures(url, session)
        candidates = []
        for capture in captures:
            ts = _utc(capture.get("timestamp"))
            if ts is not None and ts >= lower_bound:
                candidates.append((ts, capture))
        candidates.sort(key=lambda x: x[0])
        for ts, capture in candidates:
            fetched = _fetch_capture(capture, url, session)
            if not fetched:
                continue
            raw, replay_url = fetched
            try:
                keys = _keyset(raw)
            except Exception:
                continue
            if keys is not None and key in keys:
                out.at[idx, "source_available_at_utc"] = ts.isoformat()
                out.at[idx, "pit_evidence_status"] = "VERIFIED"
                out.at[idx, "pit_evidence_reason"] = f"arquivo_pt_completed_result_first_observed_after_{bound_reason.lower()}"
                out.at[idx, "pit_evidence_url"] = replay_url
                out.at[idx, "capture_digest"] = capture.get("digest")
                break
    return out
