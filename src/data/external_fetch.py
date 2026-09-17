"""Deterministic, retry-bounded HTTP fetching for external football sources.

The fetcher is deliberately source-agnostic. Source adapters are responsible for
constructing requests and parsing payloads; this layer provides common reliability,
cache, provenance and PIT metadata without ever declaring a source production-safe.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


@dataclass(frozen=True)
class FetchMetadata:
    source: str
    request_key: str
    retrieved_at: str
    http_status: int
    content_sha256: str
    content_bytes: int
    cache_hit: bool
    source_timestamp: str | None = None
    feature_available_at: str | None = None
    pit_safe: bool = False


@dataclass(frozen=True)
class CachedResponse:
    body: bytes
    metadata: FetchMetadata


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def request_key(url: str, params: dict[str, Any] | None = None) -> str:
    canonical = json.dumps({"url": url, "params": params or {}}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def content_sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def pit_is_safe(feature_available_at: str | None, prediction_cutoff_at: str | None) -> bool:
    """Return True only when both timestamps exist and availability <= cutoff."""
    if not feature_available_at or not prediction_cutoff_at:
        return False
    try:
        available = datetime.fromisoformat(feature_available_at.replace("Z", "+00:00"))
        cutoff = datetime.fromisoformat(prediction_cutoff_at.replace("Z", "+00:00"))
        if available.tzinfo is None or cutoff.tzinfo is None:
            return False
        return available.astimezone(timezone.utc) <= cutoff.astimezone(timezone.utc)
    except ValueError:
        return False


class ExternalFetcher:
    """Retry-bounded HTTP client with content-addressed raw cache."""

    def __init__(self, cache_dir: str | Path = "cache/external", timeout: float = 30.0, retries: int = 3, backoff: float = 1.5):
        self.cache_dir = Path(cache_dir)
        self.timeout = max(1.0, float(timeout))
        self.retries = max(1, int(retries))
        self.backoff = max(0.0, float(backoff))

    def _paths(self, source: str, key: str) -> tuple[Path, Path]:
        root = self.cache_dir / source
        root.mkdir(parents=True, exist_ok=True)
        return root / f"{key}.bin", root / f"{key}.json"

    def get(
        self,
        source: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        source_timestamp: str | None = None,
        feature_available_at: str | None = None,
        prediction_cutoff_at: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> CachedResponse:
        key = request_key(url, params)
        body_path, meta_path = self._paths(source, key)
        if body_path.exists() and meta_path.exists():
            try:
                body = body_path.read_bytes()
                stored = json.loads(meta_path.read_text(encoding="utf-8"))
                if stored.get("content_sha256") == content_sha256(body):
                    metadata = FetchMetadata(**stored, cache_hit=True)
                    return CachedResponse(body, metadata)
            except (OSError, ValueError, TypeError):
                pass

        last_error: Exception | None = None
        response: requests.Response | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(f"transient_http_{response.status_code}", response=response)
                response.raise_for_status()
                break
            except (requests.RequestException, OSError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(self.backoff * attempt)
        if response is None:
            raise RuntimeError(f"external fetch failed after {self.retries} attempts: {last_error}")

        body = response.content
        retrieved_at = iso_utc(utc_now())
        metadata = FetchMetadata(
            source=source,
            request_key=key,
            retrieved_at=retrieved_at,
            http_status=int(response.status_code),
            content_sha256=content_sha256(body),
            content_bytes=len(body),
            cache_hit=False,
            source_timestamp=source_timestamp,
            feature_available_at=feature_available_at,
            pit_safe=pit_is_safe(feature_available_at, prediction_cutoff_at),
        )
        try:
            body_path.write_bytes(body)
            meta_path.write_text(json.dumps(asdict(metadata), sort_keys=True, indent=2), encoding="utf-8")
        except OSError:
            # A cache failure must not turn a valid upstream response into a false data failure.
            pass
        return CachedResponse(body, metadata)
