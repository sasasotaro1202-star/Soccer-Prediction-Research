"""Shared resilient HTTP retrieval for soccer research adapters.

Network failures are expected operational events, not evidence that a source is
unavailable. This layer retries transient failures with bounded backoff and uses
a long read timeout suitable for slow public archives/datasets. It never changes
PIT semantics: callers still provide and validate their own availability timestamps.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import requests


TRANSIENT_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
DEFAULT_CONNECT_TIMEOUT = 30.0
DEFAULT_READ_TIMEOUT = 300.0
DEFAULT_RETRIES = 8
DEFAULT_BACKOFF_SECONDS = 2.0
DEFAULT_MAX_BACKOFF_SECONDS = 30.0


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        return float(default)
    return float(value) if value >= minimum else float(default)


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    return value if value >= minimum else int(default)


def default_timeout() -> tuple[float, float]:
    """Return connect/read timeouts, overridable without code changes."""
    return (
        _env_float("SOCCER_HTTP_CONNECT_TIMEOUT", DEFAULT_CONNECT_TIMEOUT, minimum=1.0),
        _env_float("SOCCER_HTTP_READ_TIMEOUT", DEFAULT_READ_TIMEOUT, minimum=5.0),
    )


def default_retries() -> int:
    return _env_int("SOCCER_HTTP_RETRIES", DEFAULT_RETRIES, minimum=1)


def _retry_delay(
    attempt: int,
    response: requests.Response | None = None,
    *,
    backoff: float | None = None,
    max_backoff: float | None = None,
) -> float:
    if response is not None:
        raw = response.headers.get("Retry-After")
        if raw:
            try:
                return min(
                    DEFAULT_MAX_BACKOFF_SECONDS if max_backoff is None else float(max_backoff),
                    max(1.0, float(raw)),
                )
            except (TypeError, ValueError):
                pass
    base = (
        _env_float("SOCCER_HTTP_BACKOFF", DEFAULT_BACKOFF_SECONDS, minimum=0.0)
        if backoff is None else max(0.0, float(backoff))
    )
    maximum = (
        _env_float("SOCCER_HTTP_MAX_BACKOFF", DEFAULT_MAX_BACKOFF_SECONDS, minimum=0.0)
        if max_backoff is None else max(0.0, float(max_backoff))
    )
    delay = min(maximum, base * (2 ** max(0, attempt - 1)))
    return max(0.0, delay)


def resilient_get(
    getter: Callable[..., requests.Response],
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float | tuple[float, float] | None = None,
    retries: int | None = None,
    backoff: float | None = None,
    max_backoff: float | None = None,
    **kwargs: Any,
) -> requests.Response:
    """GET with long read timeout and retry handling for transient failures.

    A hard non-transient HTTP error is raised immediately. Timeouts, connection
    errors, 429s and common 5xx responses are retried. The final failure is
    raised normally so callers retain fail-closed behaviour.
    """
    request_timeout = default_timeout() if timeout is None else timeout
    max_retries = default_retries() if retries is None else max(1, int(retries))
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        response: requests.Response | None = None
        try:
            response = getter(
                url,
                params=params,
                headers=headers,
                timeout=request_timeout,
                **kwargs,
            )
            if int(getattr(response, "status_code", 200)) in TRANSIENT_STATUS_CODES:
                if attempt < max_retries:
                    time.sleep(_retry_delay(attempt, response, backoff=backoff, max_backoff=max_backoff))
                    continue
            response.raise_for_status()
            return response
        except (requests.Timeout, requests.ConnectionError, requests.RequestException, OSError) as exc:
            last_error = exc
            if attempt >= max_retries:
                raise
            time.sleep(_retry_delay(attempt, response, backoff=backoff, max_backoff=max_backoff))

    if last_error is not None:
        raise last_error
    raise RuntimeError("resilient_get exhausted without a response")
