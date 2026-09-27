import json

import pytest
import requests

from src.research import safe_runner


def _patch_passing_preflight(monkeypatch):
    monkeypatch.setattr(safe_runner, "_load_gate", lambda *_args, **_kwargs: {"full_gate_passed": True})
    monkeypatch.setattr(safe_runner, "_load_audit_gate", lambda *_args, **_kwargs: {"full_gate_passed": True})
    monkeypatch.setenv("TESTS_PASSED", "true")
    monkeypatch.setenv("AUDIT_PASSED", "true")


def test_transient_engine_failure_retries_and_is_degraded(tmp_path, monkeypatch):
    out = tmp_path / "artifacts"
    monkeypatch.setenv("RESEARCH_OUTPUT_DIR", str(out))
    monkeypatch.setenv("RESEARCH_ATTEMPTS", "2")
    monkeypatch.setenv("RESEARCH_RETRY_BACKOFF", "0")
    _patch_passing_preflight(monkeypatch)

    calls = {"count": 0}

    def flaky_engine(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.exceptions.Timeout("temporary timeout")
        raise requests.exceptions.ConnectionError("connection reset")

    monkeypatch.setattr("src.research.engine.run", flaky_engine)

    assert safe_runner.run_with_retries() == 1
    payload = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    assert payload["status"] == "DEGRADED"
    assert payload["oos_claimed"] is False
    assert len(payload["errors"]) == 2
    assert payload["runner"]["status"] == "FAILED_AFTER_RETRIES"
    assert [item["retryable"] for item in payload["runner"]["retry_history"]] == [True, True]


def test_deterministic_engine_failure_fails_fast_without_retry(tmp_path, monkeypatch):
    out = tmp_path / "artifacts"
    monkeypatch.setenv("RESEARCH_OUTPUT_DIR", str(out))
    monkeypatch.setenv("RESEARCH_ATTEMPTS", "3")
    monkeypatch.setenv("RESEARCH_RETRY_BACKOFF", "0")
    _patch_passing_preflight(monkeypatch)

    calls = {"count": 0}

    def deterministic_engine(*_args, **_kwargs):
        calls["count"] += 1
        raise KeyError("home_goals")

    monkeypatch.setattr("src.research.engine.run", deterministic_engine)

    assert safe_runner.run_with_retries() == 1
    payload = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    assert calls["count"] == 1
    assert payload["status"] == "DEGRADED"
    assert payload["oos_claimed"] is False
    assert len(payload["errors"]) == 1
    assert payload["runner"]["status"] == "FAILED_FAST"
    assert payload["runner"]["retry_history"][0]["retryable"] is False


def test_preflight_block_is_nonzero_and_not_oos_claimed(tmp_path, monkeypatch):
    out = tmp_path / "artifacts"
    monkeypatch.setenv("RESEARCH_OUTPUT_DIR", str(out))
    monkeypatch.setenv("TESTS_PASSED", "false")
    monkeypatch.setenv("AUDIT_PASSED", "true")
    monkeypatch.setattr(safe_runner, "_load_gate", lambda *_args, **_kwargs: {"full_gate_passed": False, "blocking_reasons": ["gate-failed"]})
    monkeypatch.setattr(safe_runner, "_load_audit_gate", lambda *_args, **_kwargs: {"full_gate_passed": True})

    assert safe_runner.run_with_retries() == 1
    payload = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    assert payload["status"] == "BLOCKED"
    assert payload["oos_claimed"] is False
    assert payload["runner"]["exit_code"] == 1
