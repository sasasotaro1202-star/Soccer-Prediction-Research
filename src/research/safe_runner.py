from __future__ import annotations

import json
import os
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


def run_with_retries() -> int:
    """Run research only when every mandatory safety gate agrees.

    A green GitHub job is not equivalent to a valid research run. In particular,
    a completion-gate artifact cannot override a failed publication-time audit.
    Operational failures are recorded explicitly, and an exhausted engine retry
    returns a non-zero exit code so GitHub Actions can recover the failed run.
    """
    attempts = max(1, int(os.getenv("RESEARCH_ATTEMPTS", "2")))
    backoff = max(0.0, float(os.getenv("RESEARCH_RETRY_BACKOFF", "15")))
    out = Path(os.getenv("RESEARCH_OUTPUT_DIR", "artifacts"))
    out.mkdir(parents=True, exist_ok=True)

    tests_passed = os.getenv("TESTS_PASSED", "true").lower() == "true"
    audit_passed = os.getenv("AUDIT_PASSED", "true").lower() == "true"
    gate = _load_gate(out)
    audit_gate = _load_audit_gate(out)

    blockers: list[str] = []
    if not tests_passed:
        blockers.append("preflight tests failed")
    if not audit_passed:
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
        return 1

    errors: list[str] = []

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
        "runner": {"status": "STARTED"},
        "oos_claimed": False,
    })

    from src.research.engine import run

    for attempt in range(1, attempts + 1):
        try:
            report = run(str(out))
            report["runner"] = {"attempt": attempt, "max_attempts": attempts, "status": "COMPLETED"}
            _write_status(out, report)
            return 0
        except Exception as exc:
            errors.append(f"attempt={attempt} {type(exc).__name__}: {exc}")
            if attempt < attempts:
                time.sleep(backoff * attempt)

    _write_status(out, {
        "status": "DEGRADED",
        "reason": "Research engine failed after bounded retries; no OOS result was claimed.",
        "runner": {"attempts": attempts, "status": "FAILED_AFTER_RETRIES"},
        "errors": errors,
        "oos_claimed": False,
    })
    return 1


if __name__ == "__main__":
    raise SystemExit(run_with_retries())
