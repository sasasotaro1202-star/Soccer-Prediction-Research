from __future__ import annotations

import json
import math
import os
import requests
import time
from pathlib import Path


def _write_status(out: Path, payload: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_status.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def _load_json(path: Path, *, default: dict | None = None) -> dict | None:
    if not path.exists():
        return default
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else default
    except Exception as exc:
        return {
            "full_gate_passed": False,
            "blocking_reasons": [f"invalid {path.name}: {type(exc).__name__}: {exc}"],
        }


def _load_gate(out: Path) -> dict | None:
    return _load_json(out / "completion_gate.json")


def _load_audit_gate(out: Path) -> dict | None:
    return _load_json(out / "audit_gate.json")


def _read_required_bool_env(name: str) -> tuple[bool | None, str | None]:
    """Read a safety-critical boolean environment variable strictly.

    Missing or non-boolean values are not treated as passing defaults.
    """
    raw = os.getenv(name)
    if raw is None:
        return None, f"{name} is missing or invalid"
    normalized = raw.strip().lower()
    if normalized not in {"true", "false"}:
        return None, f"{name} is missing or invalid"
    return normalized == "true", None


def _is_retryable_exception(exc: Exception) -> bool:
    """Retry only failures that may plausibly clear on a bounded retry.

    Deterministic code/schema failures must fail fast to avoid wasting runner
    time and delaying the newest immutable research snapshot.
    """
    transient_types = (
        requests.exceptions.RequestException,
        TimeoutError,
        ConnectionError,
        BrokenPipeError,
    )
    return isinstance(exc, transient_types)


def _read_bounded_retry_config() -> tuple[int, float]:
    """Read retry settings without allowing malformed or unbounded values."""
    attempts_raw = os.getenv("RESEARCH_ATTEMPTS", "2").strip()
    backoff_raw = os.getenv("RESEARCH_RETRY_BACKOFF", "15").strip()
    try:
        attempts = int(attempts_raw)
    except ValueError as exc:
        raise ValueError("RESEARCH_ATTEMPTS must be an integer") from exc
    try:
        backoff = float(backoff_raw)
    except ValueError as exc:
        raise ValueError("RESEARCH_RETRY_BACKOFF must be a finite number") from exc

    # Keep the operational recovery policy bounded even when the runner is
    # invoked outside the checked-in GitHub workflow.
    if not 1 <= attempts <= 3:
        raise ValueError("RESEARCH_ATTEMPTS must be between 1 and 3")
    if not math.isfinite(backoff) or backoff < 0 or backoff > 300:
        raise ValueError("RESEARCH_RETRY_BACKOFF must be finite and between 0 and 300 seconds")
    return attempts, backoff


def run_with_retries() -> int:
    """Run research only when every mandatory safety gate agrees.

    A green GitHub job is not equivalent to a valid research run. In particular,
    a completion-gate artifact cannot override a failed publication-time audit.
    Operational failures are recorded explicitly, and an exhausted engine retry
    returns a non-zero exit code so GitHub Actions can recover the failed run.
    """
    out = Path(os.getenv("RESEARCH_OUTPUT_DIR", "artifacts"))
    out.mkdir(parents=True, exist_ok=True)

    try:
        attempts, backoff = _read_bounded_retry_config()
    except ValueError as exc:
        _write_status(out, {
            "status": "FAILED",
            "reason": "Research retry configuration is invalid; execution was fail-closed before preflight.",
            "errors": [f"{type(exc).__name__}: {exc}"],
            "retry_policy": "transient_only_bounded",
            "runner": {
                "status": "CONFIG_ERROR",
                "exit_code": 1,
            },
            "oos_claimed": False,
        })
        return 1

    tests_passed, tests_env_error = _read_required_bool_env("TESTS_PASSED")
    audit_passed, audit_env_error = _read_required_bool_env("AUDIT_PASSED")
    gate = _load_gate(out)
    audit_gate = _load_audit_gate(out)

    blockers: list[str] = []
    if tests_env_error:
        blockers.append(tests_env_error)
    elif tests_passed is False:
        blockers.append("preflight tests failed")
    if audit_env_error:
        blockers.append(audit_env_error)
    elif audit_passed is False:
        blockers.append("data audit execution failed")
    if gate is None:
        blockers.append("completion gate artifact is missing")
    elif not bool(gate.get("full_gate_passed", False)):
        blockers.extend(str(x) for x in gate.get("blocking_reasons", []) if str(x))
        if not gate.get("blocking_reasons"):
            blockers.append("completion gate did not pass")

    if audit_gate is None:
        blockers.append("audit gate artifact is missing")
    else:
        audit_full_gate = audit_gate.get("full_gate_passed")
        if audit_full_gate is not True:
            reasons = audit_gate.get("blocking_reasons") or []
            if reasons:
                blockers.extend(str(x) for x in reasons if str(x))
            else:
                blockers.append("publication-time data audit gate did not pass")

    if blockers:
        _write_status(out, {
            "status": "BLOCKED",
            "reason": "Research execution was intentionally skipped because one or more mandatory preflight gates failed or disagreed.",
            "blockers": sorted(set(blockers)),
            "gate_consistency": {
                "completion_gate_passed": bool(gate and gate.get("full_gate_passed", False)),
                "audit_gate_passed": bool(audit_gate and audit_gate.get("full_gate_passed", False)),
            },
            "runner": {"status": "SKIPPED_AFTER_PREFLIGHT_FAILURE", "exit_code": 1},
            "oos_claimed": False,
        })
        # A strict preflight block remains an explicit non-zero outcome.
        # The blocking evidence is preserved so recovery can distinguish gate rejection
        # from an engine crash without converting the failure into success.
        return 1

    errors: list[str] = []
    retry_history: list[dict] = []

    # Mark the research attempt as started before any expensive engine work.
    # This prevents stale/missing run_status artifacts from being mistaken for a
    # successful or preflight-blocked result if the runner is externally terminated.
    _write_status(out, {
        "status": "RUNNING",
        "reason": "Research engine started after mandatory preflight gates passed.",
        "gate_consistency": {
            "completion_gate_passed": True,
            "audit_gate_passed": True,
        },
        "retry_policy": "transient_only",
        "runner": {"status": "STARTED"},
        "oos_claimed": False,
    })

    from src.research.engine import run

    for attempt in range(1, attempts + 1):
        try:
            report = run(str(out))
            report["retry_policy"] = "transient_only"
            report["runner"] = {
                "attempt": attempt,
                "max_attempts": attempts,
                "status": "COMPLETED",
                "retry_history": retry_history,
            }
            _write_status(out, report)
            return 0
        except Exception as exc:
            retryable = _is_retryable_exception(exc)
            errors.append(f"attempt={attempt} {type(exc).__name__}: {exc}")
            retry_history.append(
                {
                    "attempt": attempt,
                    "exception_type": type(exc).__name__,
                    "error": str(exc),
                    "retryable": retryable,
                }
            )
            if attempt < attempts and retryable:
                time.sleep(backoff * attempt)
                continue
            break

    _write_status(out, {
        "status": "DEGRADED" if errors else "FAILED",
        "reason": "Research engine failed after a bounded retry policy; no OOS result was claimed."
        if errors
        else "Research engine did not produce a result; no OOS result was claimed.",
        "retry_policy": "transient_only",
        "retry_history": retry_history,
        "runner": {
            "attempts": len(retry_history),
            "max_attempts": attempts,
            "status": "FAILED_AFTER_RETRIES" if len(retry_history) > 1 else "FAILED_FAST",
            "retry_history": retry_history,
        },
        "errors": errors,
        "oos_claimed": False,
    })
    return 1


if __name__ == "__main__":
    raise SystemExit(run_with_retries())
