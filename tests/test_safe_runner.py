import json

from src.research import safe_runner


def test_deterministic_engine_failure_fails_fast_without_retry(tmp_path, monkeypatch):
    out = tmp_path / "artifacts"
    monkeypatch.setenv("RESEARCH_OUTPUT_DIR", str(out))
    monkeypatch.setenv("RESEARCH_ATTEMPTS", "2")
    monkeypatch.setenv("RESEARCH_RETRY_BACKOFF", "0")
    monkeypatch.setenv("TESTS_PASSED", "true")
    monkeypatch.setenv("AUDIT_PASSED", "true")
    monkeypatch.setattr(safe_runner, "_load_gate", lambda *_args, **_kwargs: {"full_gate_passed": True})
    monkeypatch.setattr(safe_runner, "_load_audit_gate", lambda *_args, **_kwargs: {"full_gate_passed": True})

    calls = {"count": 0}

    def _deterministic_failure(*_args, **_kwargs):
        calls["count"] += 1
        raise ValueError("schema mismatch")

    monkeypatch.setattr("src.research.engine.run", _deterministic_failure)

    assert safe_runner.run_with_retries() == 1
    payload = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    assert payload["status"] == "DEGRADED"
    assert payload["oos_claimed"] is False
    assert calls["count"] == 1
    assert payload["runner"]["status"] == "FAILED_FAST"
    assert payload["retry_history"][0]["retryable"] is False


def test_transient_engine_failure_retries_with_bounded_history(tmp_path, monkeypatch):
    import requests

    out = tmp_path / "artifacts"
    monkeypatch.setenv("RESEARCH_OUTPUT_DIR", str(out))
    monkeypatch.setenv("RESEARCH_ATTEMPTS", "2")
    monkeypatch.setenv("RESEARCH_RETRY_BACKOFF", "0")
    monkeypatch.setenv("TESTS_PASSED", "true")
    monkeypatch.setenv("AUDIT_PASSED", "true")
    monkeypatch.setattr(safe_runner, "_load_gate", lambda *_args, **_kwargs: {"full_gate_passed": True})
    monkeypatch.setattr(safe_runner, "_load_audit_gate", lambda *_args, **_kwargs: {"full_gate_passed": True})

    calls = {"count": 0}

    def _transient_failure(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.exceptions.Timeout("temporary timeout")
        return {"status": "COMPLETED", "oos_claimed": False}

    monkeypatch.setattr("src.research.engine.run", _transient_failure)

    assert safe_runner.run_with_retries() == 0
    payload = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    assert calls["count"] == 2
    assert payload["retry_policy"] == "transient_only_bounded"
    assert payload["runner"]["status"] == "COMPLETED"
    assert payload["runner"]["retry_history"][0]["retryable"] is True


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


def test_required_bool_env_accepts_only_explicit_booleans(monkeypatch):
    monkeypatch.setenv("TESTS_PASSED", "TRUE")
    assert safe_runner._read_required_bool_env("TESTS_PASSED") == (True, None)

    monkeypatch.setenv("TESTS_PASSED", "false")
    assert safe_runner._read_required_bool_env("TESTS_PASSED") == (False, None)

    monkeypatch.setenv("TESTS_PASSED", "1")
    value, error = safe_runner._read_required_bool_env("TESTS_PASSED")
    assert value is None
    assert error == "TESTS_PASSED is missing or invalid"


def test_invalid_retry_configuration_fails_closed_and_persists_status(tmp_path, monkeypatch):
    out = tmp_path / "artifacts"
    monkeypatch.setenv("RESEARCH_OUTPUT_DIR", str(out))
    monkeypatch.setenv("RESEARCH_ATTEMPTS", "0")
    monkeypatch.setenv("RESEARCH_RETRY_BACKOFF", "nan")
    monkeypatch.setenv("TESTS_PASSED", "true")
    monkeypatch.setenv("AUDIT_PASSED", "true")
    monkeypatch.setattr(safe_runner, "_load_gate", lambda *_args, **_kwargs: {"full_gate_passed": True})
    monkeypatch.setattr(safe_runner, "_load_audit_gate", lambda *_args, **_kwargs: {"full_gate_passed": True})

    assert safe_runner.run_with_retries() == 1
    payload = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    assert payload["status"] == "FAILED"
    assert payload["oos_claimed"] is False
    assert payload["runner"]["status"] == "CONFIG_ERROR"
    assert payload["runner"]["exit_code"] == 1


def test_retry_configuration_is_strictly_bounded(monkeypatch):
    monkeypatch.setenv("RESEARCH_ATTEMPTS", "4")
    monkeypatch.setenv("RESEARCH_RETRY_BACKOFF", "15")
    try:
        safe_runner._read_bounded_retry_config()
    except ValueError as exc:
        assert "between 1 and 3" in str(exc)
    else:
        raise AssertionError("unbounded retry attempts were accepted")

    monkeypatch.setenv("RESEARCH_ATTEMPTS", "2")
    monkeypatch.setenv("RESEARCH_RETRY_BACKOFF", "301")
    try:
        safe_runner._read_bounded_retry_config()
    except ValueError as exc:
        assert "between 0 and 300" in str(exc)
    else:
        raise AssertionError("unbounded retry backoff was accepted")
