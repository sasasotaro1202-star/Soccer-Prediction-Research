"""Research-only SofaScore official Player of the Match label adapter.

The adapter treats the label as post-match teacher information. It records when this
API response was retrieved, but never treats retrieval time as historical publication
time. Prediction features must still satisfy the project's existing PIT/OOS gates.

The endpoint used by public SofaScore implementations is
/event/{event_id}/best-players/summary and exposes playerOfTheMatch.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://api.sofascore.com/api/v1"
ENDPOINT_TEMPLATE = BASE_URL + "/event/{event_id}/best-players/summary"
HEADERS = {
    "User-Agent": "Soccer-Prediction-Research/1.0 official-POTM research",
    "Accept": "application/json",
}

def _now_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))

def _parse_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(ts) else pd.Timestamp(ts)

def parse_player_of_match(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Extract SofaScore's explicit playerOfTheMatch object without guessing."""
    if not isinstance(payload, dict):
        raise ValueError("SofaScore POTM payload root must be an object")
    raw = payload.get("playerOfTheMatch")
    if not isinstance(raw, dict):
        return None

    player = raw.get("player")
    if isinstance(player, dict):
        candidate = dict(player)
        value = raw.get("value")
        label = raw.get("label")
    else:
        candidate = dict(raw)
        value = candidate.pop("value", None)
        label = candidate.pop("label", None)

    player_id = candidate.get("id")
    player_name = candidate.get("name") or candidate.get("shortName")
    if player_id is None or not str(player_id).strip():
        raise ValueError("SofaScore POTM response has no explicit player id")
    if player_name is None or not str(player_name).strip():
        raise ValueError("SofaScore POTM response has no explicit player name")

    return {
        "player_id": str(player_id).strip(),
        "player_name": str(player_name).strip(),
        "rating_value": float(value) if value is not None else None,
        "rating_label": str(label).strip() if label is not None else "",
    }

def _get_json(
    session: requests.Session,
    event_id: str,
    *,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 1.0,
) -> tuple[dict[str, Any], pd.Timestamp]:
    url = ENDPOINT_TEMPLATE.format(event_id=str(event_id).strip())
    last_error: Exception | None = None
    for attempt in range(1, max(1, int(retries)) + 1):
        try:
            response = session.get(url, timeout=timeout, headers=HEADERS)
            response.raise_for_status()
            retrieved_at = _now_utc()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("SofaScore POTM endpoint returned a non-object JSON payload")
            return payload, retrieved_at
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(float(backoff) * attempt)
    raise RuntimeError(
        f"SofaScore POTM request failed after {max(1, int(retries))} attempts for event={event_id}"
    ) from last_error

def fetch_official_potm(
    event_id: str,
    *,
    session: requests.Session | None = None,
    timeout: float = 30.0,
    retries: int = 3,
) -> dict[str, Any]:
    eid = str(event_id).strip()
    if not eid:
        raise ValueError("event_id must not be empty")
    owned = session is None
    session = session or requests.Session()
    try:
        payload, retrieved_at = _get_json(
            session, eid, timeout=timeout, retries=retries
        )
        potm = parse_player_of_match(payload)
        return {
            "event_id": eid,
            "label_type": "SOFASCORE_OFFICIAL_POTM",
            "player_id": potm["player_id"] if potm else None,
            "player_name": potm["player_name"] if potm else None,
            "rating_value": potm["rating_value"] if potm else None,
            "rating_label": potm["rating_label"] if potm else "",
            "label_retrieved_at_utc": retrieved_at.isoformat(),
            "source_url": ENDPOINT_TEMPLATE.format(event_id=eid),
            "status": "FOUND" if potm else "NO_POTM",
            "historical_publication_time_verified": False,
        }
    finally:
        if owned:
            session.close()

def attach_labels(
    events: pd.DataFrame,
    *,
    event_id_column: str = "sofascore_event_id",
    timeout: float = 30.0,
    retries: int = 3,
) -> pd.DataFrame:
    """Attach post-match official labels to an existing event-id table."""
    if event_id_column not in events.columns:
        raise ValueError(f"missing required event id column: {event_id_column}")
    d = events.copy()
    if d[event_id_column].isna().any() or d[event_id_column].astype(str).str.strip().eq("").any():
        raise ValueError("event id column contains missing or empty values")
    if d[event_id_column].astype(str).duplicated().any():
        raise ValueError("event id column contains duplicates")

    records = []
    with requests.Session() as session:
        for event_id in d[event_id_column].astype(str):
            records.append(
                fetch_official_potm(
                    event_id,
                    session=session,
                    timeout=timeout,
                    retries=retries,
                )
            )
    labels = pd.DataFrame(records).rename(columns={"event_id": event_id_column})
    out = d.merge(labels, on=event_id_column, how="left", validate="one_to_one")
    if "kickoff_utc" in out.columns:
        kickoff = pd.to_datetime(out["kickoff_utc"], utc=True, errors="coerce")
        retrieved = pd.to_datetime(out["label_retrieved_at_utc"], utc=True, errors="coerce")
        found = out["status"].astype(str).eq("FOUND")
        invalid = found & (kickoff.isna() | retrieved.isna() | (retrieved <= kickoff))
        if bool(invalid.any()):
            raise RuntimeError(
                "Official POTM teacher was retrieved at or before kickoff; refusing unsafe label rows"
            )
    return out

def save_labels(
    events_path: str,
    *,
    output_path: str = "artifacts/sofascore_potm/official_potm_labels.csv",
    event_id_column: str = "sofascore_event_id",
) -> dict[str, Any]:
    events = pd.read_csv(events_path)
    labels = attach_labels(events, event_id_column=event_id_column)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out, index=False)
    found = int(labels["status"].eq("FOUND").sum())
    return {
        "status": "OK",
        "rows": int(len(labels)),
        "found": found,
        "missing_labels": int(len(labels) - found),
        "label_type": "SOFASCORE_OFFICIAL_POTM",
        "historical_publication_time_verified": False,
        "production_usable": False,
        "output_path": str(out),
        "generated_at_utc": _now_utc().isoformat(),
    }
