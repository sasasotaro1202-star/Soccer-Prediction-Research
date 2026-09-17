from __future__ import annotations

"""Bounded secondary PIT evidence provider using Arquivo.pt.

Audit-only provider. Matching is conservative and publication timing is never
inferred. Verification is grouped by archive URL/capture so one downloaded
snapshot can validate many historical rows without repeated network work.
"""

import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO
from urllib.parse import urlparse

import pandas as pd
import requests

from src.data.pit_source_adapter_v2 import _date_key, _row_key, _result_lower_bound

ARQUIVO_CDX = "https://arquivo.pt/wayback/cdx"
ARQUIVO_WEB = "https://arquivo.pt/wayback"
USER_AGENT = "SoccerPredictionResearch/1.2 (+PIT-audit)"


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


def _normalise_team(value):
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def _normalised_row_key(row):
    date = _date_key(row.get("Date") if hasattr(row, "get") else getattr(row, "Date", None))
    if date is None:
        date = _date_key(row.get("kickoff_utc") if hasattr(row, "get") else None)
    if date is None:
        return None
    try:
        home = row.get("HomeTeam") if hasattr(row, "get") else getattr(row, "HomeTeam", None)
        away = row.get("AwayTeam") if hasattr(row, "get") else getattr(row, "AwayTeam", None)
        hg = row.get("FTHG") if hasattr(row, "get") else getattr(row, "FTHG", None)
        ag = row.get("FTAG") if hasattr(row, "get") else getattr(row, "FTAG", None)
        result = row.get("FTR") if hasattr(row, "get") else getattr(row, "FTR", None)
        if pd.isna(hg) or pd.isna(ag) or home is None or away is None or result is None:
            return None
        return (date, _normalise_team(home), _normalise_team(away), float(hg), float(ag), str(result).strip())
    except (TypeError, ValueError):
        return None


def _history_row_key(row):
    date = _date_key(row.get("kickoff_utc"))
    if date is None:
        return None
    try:
        return (date, _normalise_team(row.get("home_team")), _normalise_team(row.get("away_team")), float(row.get("home_goals")), float(row.get("away_goals")), str(row.get("result")).strip())
    except (TypeError, ValueError):
        return _row_key(row)


def _normalise_cdx_payload(payload):
    if isinstance(payload, dict):
        for key in ("data", "results", "captures", "rows"):
            if payload.get(key) is not None:
                return _normalise_cdx_payload(payload[key])
        return []
    if not isinstance(payload, list) or not payload:
        return []
    if all(isinstance(x, dict) for x in payload):
        return [dict(x) for x in payload]
    header = payload[0]
    if not isinstance(header, (list, tuple)):
        return []
    return [dict(zip([str(x) for x in header], row)) for row in payload[1:] if isinstance(row, (list, tuple))]


def _captures(url: str, retries: int = 2, timeout: int = 10):
    params = {"url": url, "output": "json", "filter": "statuscode:200", "fl": "timestamp,original,mimetype,statuscode,digest"}
    for attempt in range(max(1, retries)):
        try:
            r = requests.get(ARQUIVO_CDX, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            return _normalise_cdx_payload(r.json())
        except (requests.RequestException, ValueError, TypeError):
            if attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1))
    return []


def _keyset(raw: bytes):
    try:
        frame = pd.read_csv(BytesIO(raw))
    except Exception:
        return None
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    if not required.issubset(frame.columns):
        return None
    return {_normalised_row_key(r) for r in frame.to_dict("records") if _normalised_row_key(r) is not None}


def _fetch_capture(capture, original_url, retries=2, timeout=10):
    ts = str(capture.get("timestamp", "")).strip()
    if not ts:
        return None
    replay_url = f"{ARQUIVO_WEB}/{ts}/{original_url}"
    for attempt in range(max(1, retries)):
        try:
            r = requests.get(replay_url, timeout=timeout, allow_redirects=True, headers={"User-Agent": USER_AGENT, "Accept": "text/csv,text/plain,*/*"})
            r.raise_for_status()
            final_url = str(getattr(r, "url", replay_url))
            parsed = urlparse(final_url)
            if parsed.hostname not in {"arquivo.pt", "www.arquivo.pt"} or not parsed.path.startswith(f"/wayback/{ts}"):
                return None
            raw = r.content
            if raw[:512].lstrip().lower().startswith((b"<!doctype html", b"<html")):
                return None
            return raw, replay_url
        except (requests.RequestException, OSError):
            if attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1))
    return None


def apply_arquivo_fallback(history: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty:
        return history.copy() if history is not None else history
    out = history.copy()
    try:
        from src.data.pit_source_adapter_v2 import source_url
        pending_by_url = {}
        for idx, row in out.iterrows():
            if str(row.get("pit_evidence_status", "")) == "VERIFIED":
                continue
            key = _history_row_key(row)
            lower_bound, bound_reason = _result_lower_bound(row)
            if key is None or lower_bound is None:
                continue
            try:
                url = source_url(str(row.get("competition")), int(row.get("season_start")))
            except (TypeError, ValueError, KeyError):
                continue
            pending_by_url.setdefault(url, []).append((idx, key, lower_bound, bound_reason))

        def verify_url(url, rows):
            unresolved = {key: (idx, bound, reason) for idx, key, bound, reason in rows}
            if not unresolved:
                return []
            captures = _captures(url)
            candidates = []
            for capture in captures:
                ts = _utc(capture.get("timestamp"))
                if ts is not None and any(ts >= item[1] for item in unresolved.values()):
                    candidates.append((ts, capture))
            candidates.sort(key=lambda x: x[0])
            found = []
            for ts, capture in candidates:
                if not unresolved:
                    break
                fetched = _fetch_capture(capture, url)
                if not fetched:
                    continue
                raw, replay_url = fetched
                keys = _keyset(raw)
                if keys is None:
                    continue
                for key in list(unresolved):
                    idx, bound, reason = unresolved[key]
                    if ts >= bound and key in keys:
                        found.append((idx, ts, replay_url, capture, reason))
                        del unresolved[key]
            return found

        max_workers = min(4, max(1, len(pending_by_url)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(verify_url, url, rows): url for url, rows in pending_by_url.items()}
            for future in as_completed(futures):
                try:
                    results = future.result()
                except Exception:
                    results = []
                for idx, ts, replay_url, capture, bound_reason in results:
                    out.at[idx, "source_available_at_utc"] = ts.isoformat()
                    out.at[idx, "pit_evidence_status"] = "VERIFIED"
                    out.at[idx, "pit_evidence_reason"] = f"arquivo_pt_completed_result_first_observed_after_{bound_reason.lower()}"
                    out.at[idx, "pit_evidence_url"] = replay_url
                    out.at[idx, "capture_digest"] = capture.get("digest")
    except Exception:
        # Audit failure leaves rows unchanged; it never corrupts the main dataset.
        pass
    return out
