from __future__ import annotations

"""Bounded secondary PIT evidence provider using Arquivo.pt.

This module is an audit provider only. It must never block the main research
engine when an archive service is unavailable. Team matching is deliberately
conservative: Unicode/diacritic normalization and punctuation folding only;
no semantic aliases are introduced.
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
USER_AGENT = "SoccerPredictionResearch/1.1 (+PIT-audit)"


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


def _normalise_team(value) -> str:
    """Conservative identity normalization; intentionally no semantic aliases."""
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def _normalised_row_key(row) -> tuple | None:
    """Build the same conservative archive key used by the primary PIT adapter."""
    date = _date_key(row.get("Date") if hasattr(row, "get") else getattr(row, "Date", None))
    if date is None:
        kickoff = row.get("kickoff_utc") if hasattr(row, "get") else None
        date = _date_key(kickoff)
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
        return (
            date,
            _normalise_team(row.get("home_team")),
            _normalise_team(row.get("away_team")),
            float(row.get("home_goals")),
            float(row.get("away_goals")),
            str(row.get("result")).strip(),
        )
    except (TypeError, ValueError):
        # Fall back to the canonical primary-adapter key if available.
        return _row_key(row)


def _normalise_cdx_payload(payload):
    """Accept list-of-lists, list-of-dicts and mapping-wrapped CDX responses."""
    if isinstance(payload, dict):
        for key in ("data", "results", "captures", "rows"):
            value = payload.get(key)
            if value is not None:
                return _normalise_cdx_payload(value)
        return []
    if not isinstance(payload, list) or not payload:
        return []
    if all(isinstance(x, dict) for x in payload):
        return [dict(x) for x in payload]
    header = payload[0]
    if not isinstance(header, (list, tuple)):
        return []
    names = [str(x) for x in header]
    return [dict(zip(names, row)) for row in payload[1:] if isinstance(row, (list, tuple))]


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
    keys = set()
    for r in frame.to_dict("records"):
        key = _normalised_row_key(r)
        if key is not None:
            keys.add(key)
    return keys


def _fetch_capture(capture: dict, original_url: str, retries: int = 2, timeout: int = 10):
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
        pending = []
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
            pending.append((idx, key, lower_bound, bound_reason, url))

        urls = sorted({x[4] for x in pending})
        capture_map = {}
        with ThreadPoolExecutor(max_workers=min(4, max(1, len(urls)))) as pool:
            futures = {pool.submit(_captures, url): url for url in urls}
            for future in as_completed(futures):
                try:
                    capture_map[futures[future]] = future.result()
                except Exception:
                    capture_map[futures[future]] = []

        jobs = []
        for idx, key, lower_bound, bound_reason, url in pending:
            plausible = []
            for capture in capture_map.get(url, []):
                ts = _utc(capture.get("timestamp"))
                if ts is not None and ts >= lower_bound:
                    plausible.append((ts, capture))
            plausible.sort(key=lambda x: x[0])
            jobs.append((idx, key, bound_reason, url, plausible[:4]))

        def verify(job):
            idx, key, bound_reason, url, captures = job
            for ts, capture in captures:
                fetched = _fetch_capture(capture, url)
                if not fetched:
                    continue
                raw, replay_url = fetched
                keys = _keyset(raw)
                if keys is not None and key in keys:
                    return idx, ts, replay_url, capture, bound_reason
            return None

        with ThreadPoolExecutor(max_workers=min(4, max(1, len(jobs)))) as pool:
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
        # Audit failure is represented by unchanged rows; never corrupt the main dataset.
        pass
    return out
