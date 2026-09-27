"""Immutable source snapshots for point-in-time soccer research.

A snapshot can be retrieved after a historical cutoff and still be stored.
Eligibility is decided separately by available_at <= prediction_cutoff and an
explicit KNOWN status. Existing records are never rewritten.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any


ALLOWED_STATUS = {"KNOWN", "MISSING", "UNAVAILABLE", "UNVERIFIABLE"}


def _dt(value: str) -> datetime:
    ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if ts.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return ts


def payload_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceSnapshot:
    source_key: str
    event_id: str
    entity_id: str
    source_timestamp: str | None
    retrieved_at: str
    available_at: str | None
    prediction_cutoff: str
    payload_hash: str
    status: str = "KNOWN"

    def validate(self) -> None:
        if not self.source_key or not self.event_id or not self.entity_id:
            raise ValueError("source_key, event_id and entity_id are required")
        if self.status not in ALLOWED_STATUS:
            raise ValueError(f"invalid snapshot status: {self.status}")
        _dt(self.retrieved_at)
        cutoff = _dt(self.prediction_cutoff)
        if self.source_timestamp:
            _dt(self.source_timestamp)
        if self.available_at:
            available = _dt(self.available_at)
            retrieved = _dt(self.retrieved_at)
            if available > retrieved:
                raise ValueError("available_at cannot be after retrieved_at")
            # A future availability relative to cutoff is allowed to be stored;
            # eligibility() will reject it. This preserves full provenance.


def make_snapshot(
    *,
    source_key: str,
    event_id: str,
    entity_id: str,
    payload: Any,
    prediction_cutoff: str,
    available_at: str | None,
    source_timestamp: str | None = None,
    retrieved_at: str | None = None,
    status: str = "KNOWN",
) -> SourceSnapshot:
    from datetime import datetime, timezone
    snapshot = SourceSnapshot(
        source_key=str(source_key),
        event_id=str(event_id),
        entity_id=str(entity_id),
        source_timestamp=source_timestamp,
        retrieved_at=retrieved_at or datetime.now(timezone.utc).isoformat(),
        available_at=available_at,
        prediction_cutoff=prediction_cutoff,
        payload_hash=payload_hash(payload),
        status=str(status).upper(),
    )
    snapshot.validate()
    return snapshot


def pit_eligible(snapshot: SourceSnapshot) -> bool:
    snapshot.validate()
    if snapshot.status != "KNOWN" or not snapshot.available_at:
        return False
    return _dt(snapshot.available_at) <= _dt(snapshot.prediction_cutoff)


def append_snapshot(snapshot: SourceSnapshot, path: str | Path) -> None:
    snapshot.validate()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Append-only by design. A later observation with the same event/entity is
    # a new provenance record, not an overwrite of historical evidence.
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(snapshot), ensure_ascii=False, sort_keys=True) + "\n")


def read_snapshots(path: str | Path) -> list[SourceSnapshot]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        snap = SourceSnapshot(**data)
        snap.validate()
        out.append(snap)
    return out
