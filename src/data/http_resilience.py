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
DEFAULT_CONNECT_TIMEOUT = 60.0
DEFAULT_READ_TIMEOUT = 600.0
MIN_CONNECT_TIMEOUT = 30.0
MIN_READ_TIMEOUT = 600.0
DEFAULT_RETRIES = 8
DEFAULT_BACKOFF_SECONDS = 2.0
DEFAULT_MAX_BACKOFF_SECONDS = 60.0

# Chat/interactive callers must not inherit the long batch/archive budget. They
# can opt into this bounded profile explicitly while batch workflows keep the
# existing long-read defaults.
INTERACTIVE_CONNECT_TIMEOUT = 15.0
INTERACTIVE_READ_TIMEOUT = 30.0
INTERACTIVE_RETRIES = 2
INTERACTIVE_MAX_BACKOFF_SECONDS = 4.0


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
    """Return connect/read timeouts, overridable only above the safety floor."""
    return (
        _env_float(
            "SOCCER_HTTP_CONNECT_TIMEOUT",
            DEFAULT_CONNECT_TIMEOUT,
            minimum=MIN_CONNECT_TIMEOUT,
        ),
        _env_float(
            "SOCCER_HTTP_READ_TIMEOUT",
            DEFAULT_READ_TIMEOUT,
            minimum=MIN_READ_TIMEOUT,
        ),
    )


def default_retries(*, profile: str = "batch") -> int:
    if str(profile).strip().lower() == "interactive":
        return _env_int("SOCCER_HTTP_INTERACTIVE_RETRIES", INTERACTIVE_RETRIES, minimum=1)
    return _env_int("SOCCER_HTTP_RETRIES", DEFAULT_RETRIES, minimum=1)


def profile_defaults(profile: str = "batch") -> tuple[tuple[float, float], int, float]:
    """Return timeout/retry/backoff defaults for batch or interactive callers."""
    if str(profile).strip().lower() == "interactive":
        return (
            (INTERACTIVE_CONNECT_TIMEOUT, INTERACTIVE_READ_TIMEOUT),
            default_retries(profile="interactive"),
            INTERACTIVE_MAX_BACKOFF_SECONDS,
        )
    return (default_timeout(), default_retries(profile="batch"), DEFAULT_MAX_BACKOFF_SECONDS)


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
    profile: str = "batch",
    deadline_seconds: float | None = None,
    **kwargs: Any,
) -> requests.Response:
    """GET with long read timeout and retry handling for transient failures.

    A hard non-transient HTTP error is raised immediately. Timeouts, connection
    errors, 429s and common 5xx responses are retried. The final failure is
    raised normally so callers retain fail-closed behaviour.
    """
    profile_name = str(profile).strip().lower()
    profile_timeout, profile_retries, profile_max_backoff = profile_defaults(profile_name)
    if timeout is None:
        request_timeout = profile_timeout
    elif isinstance(timeout, tuple):
        request_timeout = (
            max(MIN_CONNECT_TIMEOUT, float(timeout[0])),
            max(MIN_READ_TIMEOUT, float(timeout[1])),
        )
    else:
        # Do not allow legacy callers to silently reintroduce very short
        # socket/read timeouts. A caller may increase this, never reduce it.
        value = float(timeout)
        request_timeout = (
            max(MIN_CONNECT_TIMEOUT, value),
            max(MIN_READ_TIMEOUT, value),
        )
    max_retries = profile_retries if retries is None else max(1, int(retries))
    last_error: Exception | None = None
    started = time.monotonic()

    for attempt in range(1, max_retries + 1):
        remaining = None
        if deadline_seconds is not None:
            remaining = max(0.0, float(deadline_seconds) - (time.monotonic() - started))
            if remaining <= 0.0:
                raise TimeoutError("resilient_get deadline exhausted before request attempt")
        effective_timeout = request_timeout
        # A caller-supplied deadline is a hard wall for the entire operation.
        # Therefore a single socket/read timeout must never extend beyond the
        # remaining interactive budget.
        if remaining is not None:
            effective_timeout = (
                min(float(request_timeout[0]), remaining),
                min(float(request_timeout[1]), remaining),
            )
        response: requests.Response | None = None
        try:
            response = getter(
                url,
                params=params,
                headers=headers,
                timeout=effective_timeout,
                **kwargs,
            )
        except (requests.Timeout, requests.ConnectionError, requests.RequestException, OSError) as exc:
            last_error = exc
            if attempt >= max_retries:
                raise
            delay = _retry_delay(attempt, None, backoff=backoff, max_backoff=(profile_max_backoff if max_backoff is None else max_backoff))
            if deadline_seconds is not None and time.monotonic() - started + delay >= max(0.0, float(deadline_seconds)):
                raise TimeoutError("resilient_get deadline exhausted before retry") from exc
            time.sleep(delay)
            continue

        status_code = int(getattr(response, "status_code", 200))
        if status_code in TRANSIENT_STATUS_CODES:
            if attempt < max_retries:
                # Release the transient response before sleeping/retrying so repeated
                # 429/5xx responses cannot accumulate open connection resources.
                response.close()
                delay = _retry_delay(attempt, response, backoff=backoff, max_backoff=(profile_max_backoff if max_backoff is None else max_backoff))
                if deadline_seconds is not None and time.monotonic() - started + delay >= max(0.0, float(deadline_seconds)):
                    raise TimeoutError("resilient_get deadline exhausted before retry")
                time.sleep(delay)
                continue

        # Permanent HTTP errors are source-state signals, not transient network failures.
        response.raise_for_status()
        return response

    if last_error is not None:
        raise last_error
    raise RuntimeError("resilient_get exhausted without a response")
