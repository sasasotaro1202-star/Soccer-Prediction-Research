"""PIT-safe normalization for social-media evidence.

This module intentionally does not scrape or infer social content. It accepts raw
exports/API payloads and normalizes only fields needed for point-in-time auditing.
A record is production-eligible only when publication/availability is explicit and
is no later than the prediction cutoff. Edit/deletion uncertainty is fail-closed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class SocialEvidence:
    source: str
    post_id: str
    author_id: str | None
    author_handle: str | None
    published_at_utc: str | None
    edited_at_utc: str | None
    retrieved_at_utc: str
    prediction_cutoff_at_utc: str | None
    text: str
    raw_sha256: str
    pit_safe: bool
    pit_reason: str


def _parse_ts(value: Any) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None
    return dt.astimezone(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat().replace("+00:00", "Z") if dt else None


def _canonical_raw(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")


def _extract(payload: dict[str, Any], source: str) -> tuple[str | None, str | None, str | None, str | None, str]:
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        raise ValueError("social payload must contain an object")
    if source == "x_api":
        post_id = data.get("id")
        author_id = data.get("author_id")
        handle = data.get("username")
        published = data.get("created_at")
        edited = data.get("edited_at")
        text = data.get("text", "")
    else:
        post_id = data.get("id")
        author_id = data.get("from", {}).get("id") if isinstance(data.get("from"), dict) else data.get("author_id")
        handle = data.get("username") or data.get("from", {}).get("name") if isinstance(data.get("from"), dict) else data.get("username")
        published = data.get("timestamp") or data.get("created_time") or data.get("created_at")
        edited = data.get("updated_time") or data.get("edited_at")
        text = data.get("caption") or data.get("message") or data.get("text") or ""
    return (
        str(post_id) if post_id is not None else None,
        str(author_id) if author_id is not None else None,
        str(handle) if handle is not None else None,
        str(published) if published is not None else None,
        str(text),
    )


def normalize_social_evidence(
    source: str,
    payload: dict[str, Any],
    *,
    retrieved_at_utc: str,
    prediction_cutoff_at_utc: str,
) -> SocialEvidence:
    allowed = {"x_api", "instagram_graph_api", "facebook_graph_api"}
    if source not in allowed:
        raise ValueError(f"unsupported social source: {source}")
    post_id, author_id, handle, published_raw, text = _extract(payload, source)
    if not post_id:
        raise ValueError("social payload missing post id")
    published = _parse_ts(published_raw)
    cutoff = _parse_ts(prediction_cutoff_at_utc)
    retrieved = _parse_ts(retrieved_at_utc)
    if retrieved is None or cutoff is None:
        raise ValueError("retrieved_at_utc and prediction_cutoff_at_utc must be timezone-aware ISO timestamps")

    # Edit timestamps are treated conservatively: if an edit is known to have
    # occurred after cutoff, the record is not usable for that prediction.
    data = payload.get("data", payload)
    edited_raw = data.get("edited_at") or data.get("updated_time") or data.get("edited_time")
    edited = _parse_ts(edited_raw)
    if published is None:
        pit_safe, reason = False, "publication_time_missing_or_invalid"
    elif published > cutoff:
        pit_safe, reason = False, "published_after_prediction_cutoff"
    elif edited is not None and edited > cutoff:
        pit_safe, reason = False, "content_edited_after_prediction_cutoff"
    elif published > retrieved:
        pit_safe, reason = False, "publication_after_retrieval_time"
    else:
        pit_safe, reason = True, "explicit_publication_time_before_cutoff"

    return SocialEvidence(
        source=source,
        post_id=post_id,
        author_id=author_id,
        author_handle=handle,
        published_at_utc=_iso(published),
        edited_at_utc=_iso(edited),
        retrieved_at_utc=_iso(retrieved) or "",
        prediction_cutoff_at_utc=_iso(cutoff),
        text=text,
        raw_sha256=hashlib.sha256(_canonical_raw(payload)).hexdigest(),
        pit_safe=pit_safe,
        pit_reason=reason,
    )
