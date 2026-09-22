import json

from src.research import safe_runner


def test_engine_failure_after_retries_is_nonzero_and_degraded(tmp_path, monkeypatch):
    out = tmp_path / "artifacts"
    monkeypatch.setenv("RESEARCH_OUTPUT_DIR", str(out))
    monkeypatch.setenv("RESEARCH_ATTEMPTS", "2")
    monkeypatch.setenv("RESEARCH_RETRY_BACKOFF", "0")
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
