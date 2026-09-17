"""Append-only PIT snapshot storage for research-only external data.

This module is deliberately independent from the incumbent production model.
It stores raw payloads plus immutable provenance and exposes a strict cutoff
query.  A snapshot is usable only when its feature-availability timestamp is
known and is at or before the prediction cutoff.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _parse_time(value: str) -> datetime:
    value = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PITSnapshot:
    source: str
    request_key: str
    entity_key: str
    captured_at: str
    feature_available_at: str | None
    source_timestamp: str | None
    prediction_cutoff_at: str | None
    content_sha256: str
    payload: Any

    @property
    def pit_safe(self) -> bool:
        if not self.feature_available_at or not self.prediction_cutoff_at:
            return False
        return _parse_time(self.feature_available_at) <= _parse_time(self.prediction_cutoff_at)

    def to_json(self) -> str:
        return canonical_json(asdict(self))


class PITSnapshotStore:
    """Content-addressed, append-only JSONL store.

    Duplicate request/entity/content combinations are ignored, making repeated
    scheduled runs idempotent.  Existing records are never overwritten.
    """

    def __init__(self, root: str | Path = "cache/xdata_pit") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "snapshots.jsonl"
        self.index_path = self.root / "index.json"

    def _load_index(self) -> set[str]:
        if not self.index_path.exists():
            return set()
        try:
            return set(json.loads(self.index_path.read_text("utf-8")))
        except (OSError, ValueError, TypeError):
            # A corrupt index must not silently suppress writes.
            return set()

    def put(self, snapshot: PITSnapshot) -> bool:
        key = sha256_text(
            canonical_json({
                "source": snapshot.source,
                "request_key": snapshot.request_key,
                "entity_key": snapshot.entity_key,
                "content_sha256": snapshot.content_sha256,
            })
        )
        index = self._load_index()
        if key in index:
            return False
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(snapshot.to_json() + "\n")
        index.add(key)
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(canonical_json(sorted(index)), encoding="utf-8")
        tmp.replace(self.index_path)
        return True

    def iter_snapshots(self) -> Iterable[PITSnapshot]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                yield PITSnapshot(**json.loads(line))

    def query_pit_safe(self, *, source: str, entity_key: str, cutoff_at: str) -> list[PITSnapshot]:
        cutoff = _parse_time(cutoff_at)
        rows = []
        for row in self.iter_snapshots():
            if row.source != source or row.entity_key != entity_key:
                continue
            if not row.feature_available_at:
                continue
            if _parse_time(row.feature_available_at) <= cutoff:
                rows.append(row)
        rows.sort(key=lambda r: _parse_time(r.feature_available_at or r.captured_at), reverse=True)
        return rows


def build_snapshot(*, source: str, request_key: str, entity_key: str, payload: Any,
                   captured_at: str, feature_available_at: str | None = None,
                   source_timestamp: str | None = None,
                   prediction_cutoff_at: str | None = None) -> PITSnapshot:
    return PITSnapshot(
        source=source,
        request_key=request_key,
        entity_key=entity_key,
        captured_at=captured_at,
        feature_available_at=feature_available_at,
        source_timestamp=source_timestamp,
        prediction_cutoff_at=prediction_cutoff_at,
        content_sha256=sha256_text(canonical_json(payload)),
        payload=payload,
    )
