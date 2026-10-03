import json

from src.research import safe_runner


def test_engine_failure_after_retries_is_nonzero_and_degraded(tmp_path, monkeypatch):
    out = tmp_path / "artifacts"
    monkeypatch.setenv("RESEARCH_OUTPUT_DIR", str(out))
    monkeypatch.setenv("RESEARCH_ATTEMPTS", "2")
    monkeypatch.setenv("RESEARCH_RETRY_BACKOFF", "0")
    monkeypatch.setenv("TESTS_PASSED", "true")
    monkeypatch.setenv("AUDIT_PASSED", "true")
    monkeypatch.setattr(safe_runner, "_load_gate", lambda *_args, **_kwargs: {"full_gate_passed": True})
    monkeypatch.setattr(safe_runner, "_load_audit_gate", lambda *_args, **_kwargs: {"full_gate_passed": True})
    monkeypatch.setattr(safe_runner, "_load_gate", lambda *_args, **_kwargs: {"full_gate_passed": True})
    monkeypatch.setattr(safe_runner, "_load_audit_gate", lambda *_args, **_kwargs: {"full_gate_passed": True})
    monkeypatch.setattr(safe_runner, "_load_gate", lambda *_args, **_kwargs: {"full_gate_passed": True})
    monkeypatch.setattr(safe_runner, "_load_audit_gate", lambda *_args, **_kwargs: {"full_gate_passed": True})

    monkeypatch.setattr(
        "src.research.engine.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("engine exploded")),
    )

    assert safe_runner.run_with_retries() == 1
    payload = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    assert payload["status"] == "DEGRADED"
    assert payload["oos_claimed"] is False
    assert len(payload["errors"]) == 2


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


def test_missing_preflight_env_fails_closed(tmp_path, monkeypatch):
    out = tmp_path / "artifacts"
    monkeypatch.setenv("RESEARCH_OUTPUT_DIR", str(out))
    monkeypatch.delenv("TESTS_PASSED", raising=False)
    monkeypatch.delenv("AUDIT_PASSED", raising=False)
    monkeypatch.setattr(safe_runner, "_load_gate", lambda *_args, **_kwargs: {"full_gate_passed": True})
    monkeypatch.setattr(safe_runner, "_load_audit_gate", lambda *_args, **_kwargs: {"full_gate_passed": True})

    def _unexpected_engine_call(*_args, **_kwargs):
        raise AssertionError("research engine must not run when preflight environment is missing")

    monkeypatch.setattr("src.research.engine.run", _unexpected_engine_call)

    assert safe_runner.run_with_retries() == 1
    payload = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    assert payload["status"] == "BLOCKED"
    assert payload["oos_claimed"] is False
    assert "TESTS_PASSED is missing or invalid" in payload["blockers"]
    assert "AUDIT_PASSED is missing or invalid" in payload["blockers"]


def test_invalid_preflight_env_fails_closed(tmp_path, monkeypatch):
    out = tmp_path / "artifacts"
    monkeypatch.setenv("RESEARCH_OUTPUT_DIR", str(out))
    monkeypatch.setenv("TESTS_PASSED", "maybe")
    monkeypatch.setenv("AUDIT_PASSED", "true")
    monkeypatch.setattr(safe_runner, "_load_gate", lambda *_args, **_kwargs: {"full_gate_passed": True})
    monkeypatch.setattr(safe_runner, "_load_audit_gate", lambda *_args, **_kwargs: {"full_gate_passed": True})

    def _unexpected_engine_call(*_args, **_kwargs):
        raise AssertionError("research engine must not run when preflight environment is invalid")

    monkeypatch.setattr("src.research.engine.run", _unexpected_engine_call)

    assert safe_runner.run_with_retries() == 1
    payload = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    assert payload["status"] == "BLOCKED"
    assert "TESTS_PASSED is missing or invalid" in payload["blockers"]
