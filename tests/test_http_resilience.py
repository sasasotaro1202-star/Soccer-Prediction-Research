from __future__ import annotations

from pathlib import Path

import requests

from src.data import http_resilience


def test_default_http_profile_is_interactive_outside_actions(monkeypatch):
    monkeypatch.delenv("SOCCER_HTTP_PROFILE", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert http_resilience.default_profile() == "interactive"


def test_default_http_profile_is_batch_in_actions(monkeypatch):
    monkeypatch.delenv("SOCCER_HTTP_PROFILE", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert http_resilience.default_profile() == "batch"


def test_explicit_http_profile_overrides_auto_detection(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "false")
    monkeypatch.setenv("SOCCER_HTTP_PROFILE", "batch")
    assert http_resilience.default_profile() == "batch"


def test_default_http_timeout_is_long_read_timeout():
    assert http_resilience.default_timeout() == (60.0, 600.0)
    assert http_resilience.default_retries() == 8


def test_interactive_get_uses_bounded_profile(monkeypatch):
    observed = {}

    class Response:
        status_code = 200
        headers = {}
        def raise_for_status(self):
            return None

    def fake_get(*args, **kwargs):
        observed.update(kwargs)
        return Response()

    monkeypatch.setattr(http_resilience.requests, "get", fake_get)
    response = http_resilience.interactive_get(
        "https://example.test/chat-safe",
        deadline_seconds=12,
    )
    assert response.status_code == 200
    assert all(abs(value - 12.0) < 1e-3 for value in observed["timeout"])
    assert observed["headers"] is None


def test_interactive_profile_is_bounded():
    timeout, retries, max_backoff = http_resilience.profile_defaults("interactive")
    assert timeout == (15.0, 30.0)
    assert retries == 2
    assert max_backoff == 4.0


def test_resilient_get_interactive_profile_uses_short_budget(monkeypatch):
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
        "https://example.test/interactive",
        profile="interactive",
        retries=1,
        deadline_seconds=5,
    )
    assert all(abs(value - 5.0) < 1e-3 for value in observed["timeout"])


def test_resilient_get_clamps_request_timeout_to_remaining_deadline(monkeypatch):
    observed = {}

    class Response:
        status_code = 200
        headers = {}
        def raise_for_status(self):
            return None

    monotonic_values = iter([0.0, 0.0, 4.0])
    monkeypatch.setattr(http_resilience.time, "monotonic", lambda: next(monotonic_values))

    def fake_get(*args, **kwargs):
        observed["timeout"] = kwargs["timeout"]
        return Response()

    http_resilience.resilient_get(
        fake_get,
        "https://example.test/deadline-clamp",
        profile="interactive",
        retries=1,
        deadline_seconds=4,
    )
    assert observed["timeout"] == (4.0, 4.0)


def test_resilient_get_rejects_retry_when_interactive_deadline_is_exhausted(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(http_resilience.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(http_resilience.time, "sleep", lambda _: None)

    def fake_get(*args, **kwargs):
        calls["n"] += 1
        raise requests.Timeout("temporary")

    try:
        http_resilience.resilient_get(
            fake_get,
            "https://example.test/deadline",
            profile="interactive",
            retries=2,
            deadline_seconds=1,
            backoff=2,
        )
    except TimeoutError:
        pass
    else:
        raise AssertionError("interactive deadline must fail fast rather than retry indefinitely")
    assert calls["n"] == 1


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
        profile="batch",
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
        def close(self):
            return None
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


def test_default_http_timeout_clamps_short_environment_overrides(monkeypatch):
    monkeypatch.setenv("SOCCER_HTTP_CONNECT_TIMEOUT", "1")
    monkeypatch.setenv("SOCCER_HTTP_READ_TIMEOUT", "10")
    assert http_resilience.default_timeout() == (30.0, 600.0)


def test_default_http_timeout_rejects_nonfinite_environment_overrides(monkeypatch):
    monkeypatch.setenv("SOCCER_HTTP_CONNECT_TIMEOUT", "nan")
    monkeypatch.setenv("SOCCER_HTTP_READ_TIMEOUT", "inf")
    assert http_resilience.default_timeout() == (60.0, 600.0)


def test_9h_watchdog_bounds_all_control_plane_http_calls():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "soccer-9h-queue-watchdog.yml"
    )
    text = workflow.read_text(encoding="utf-8")
    assert "timeout-minutes: 10" in text
    assert "CURL_TIMEOUT_ARGS=(--connect-timeout 30 --max-time 120)" in text
    assert text.count('"${CURL_TIMEOUT_ARGS[@]}"') == 4


def test_football_data_uses_shared_http_resilience():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "data"
        / "football_data.py"
    ).read_text(encoding="utf-8")
    assert "from src.data.http_resilience import resilient_get" in source
    assert "return resilient_get(" in source
    assert "timeout=(30.0, 300.0)" not in source
    assert "session.get(" not in source


def test_resilient_get_closes_transient_response_before_retry(monkeypatch):
    calls = {"n": 0}
    closed = []

    class Response:
        def __init__(self, status):
            self.status_code = status
            self.headers = {}
        def close(self):
            closed.append(self.status_code)
        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError("transient", response=self)

    def fake_get(*args, **kwargs):
        calls["n"] += 1
        return Response(503 if calls["n"] == 1 else 200)

    monkeypatch.setattr(http_resilience.time, "sleep", lambda seconds: None)
    response = http_resilience.resilient_get(fake_get, "https://example.test/close", retries=2)

    assert response.status_code == 200
    assert closed == [503]

def test_interactive_explicit_timeout_uses_interactive_floor(monkeypatch):
    monkeypatch.delenv("SOCCER_HTTP_PROFILE", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    calls = []

    class Response:
        status_code = 200
        headers = {}
        def raise_for_status(self):
            return None
        def close(self):
            return None

    def fake_get(*args, **kwargs):
        calls.append(kwargs["timeout"])
        return Response()

    http_resilience.resilient_get(
        fake_get,
        "https://example.test/data",
        timeout=(5, 10),
        retries=1,
        profile="interactive",
    )
    assert calls == [(15.0, 30.0)]

def test_live_http_profile_is_tightly_bounded():
    timeout, retries, max_backoff = http_resilience.profile_defaults("live")
    assert timeout == (20.0, 45.0)
    assert retries == 3
    assert max_backoff == 4.0


def test_live_profile_uses_live_timeout_floor():
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
        "https://example.test/live",
        profile="live",
        timeout=(1, 1),
        retries=1,
    )
    assert observed["timeout"] == (15.0, 45.0)


def test_matchday_fetcher_selects_live_http_profile():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "data"
        / "matchday_intelligence_fetch.py"
    ).read_text(encoding="utf-8")
    assert 'profile="live"' in source
    assert "retries=3" in source
    assert "backoff=1.0" in source
