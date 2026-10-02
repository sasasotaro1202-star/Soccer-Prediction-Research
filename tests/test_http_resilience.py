from __future__ import annotations

import requests

from src.data import http_resilience


def test_default_http_timeout_is_long_read_timeout():
    assert http_resilience.default_timeout() == (60.0, 600.0)
    assert http_resilience.default_retries() == 8


def test_resilient_get_retries_timeout_then_succeeds(monkeypatch):
    calls = {"n": 0}
    sleeps = []

    class Response:
        status_code = 200
        headers = {}
        def raise_for_status(self):
            return None

    def fake_get(*args, **kwargs):
        calls["n"] += 1
        assert kwargs["timeout"] == (60.0, 600.0)
        if calls["n"] < 3:
            raise requests.Timeout("temporary read timeout")
        return Response()

    monkeypatch.setattr(http_resilience.time, "sleep", lambda seconds: sleeps.append(seconds))
    response = http_resilience.resilient_get(
        fake_get,
        "https://example.test/data",
        retries=3,
        backoff=2.0,
    )

    assert response.status_code == 200
    assert calls["n"] == 3
    assert sleeps == [2.0, 4.0]


def test_resilient_get_retries_transient_http_status_and_respects_retry_after(monkeypatch):
    calls = {"n": 0}
    sleeps = []

    class Response:
        def __init__(self, status):
            self.status_code = status
            self.headers = {"Retry-After": "3"} if status == 503 else {}
        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def fake_get(*args, **kwargs):
        calls["n"] += 1
        return Response(503 if calls["n"] < 2 else 200)

    monkeypatch.setattr(http_resilience.time, "sleep", lambda seconds: sleeps.append(seconds))
    response = http_resilience.resilient_get(
        fake_get,
        "https://example.test/data",
        retries=2,
        backoff=2.0,
    )

    assert response.status_code == 200
    assert calls["n"] == 2
    assert sleeps == [3.0]


def test_resilient_get_does_not_retry_non_transient_http_error(monkeypatch):
    calls = {"n": 0}

    class Response:
        status_code = 404
        headers = {}
        def raise_for_status(self):
            raise requests.HTTPError("HTTP 404", response=self)

    def fake_get(*args, **kwargs):
        calls["n"] += 1
        return Response()

    try:
        http_resilience.resilient_get(
            fake_get,
            "https://example.test/missing",
            retries=8,
        )
    except requests.HTTPError:
        pass
    else:
        raise AssertionError("non-transient HTTP errors must remain hard failures")

    assert calls["n"] == 1


def test_resilient_get_clamps_legacy_short_timeouts(monkeypatch):
    observed = {}

    class Response:
        status_code = 200
        headers = {}
        def raise_for_status(self):
            return None

    def fake_get(*args, **kwargs):
        observed["timeout"] = kwargs["timeout"]
        return Response()

    http_resilience.resilient_get(
        fake_get,
        "https://example.test/legacy",
        timeout=10,
        retries=1,
    )
    assert observed["timeout"] == (30.0, 600.0)

    http_resilience.resilient_get(
        fake_get,
        "https://example.test/legacy-tuple",
        timeout=(1, 20),
        retries=1,
    )
    assert observed["timeout"] == (30.0, 600.0)
