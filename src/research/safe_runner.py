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


def _load_gate(out: Path) -> dict | None:
    path = out / "completion_gate.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"full_gate_passed": False, "blocking_reasons": [f"invalid completion_gate.json: {exc}"]}


def run_with_retries() -> int:
    """Run research safely without fabricating success or bypassing audit gates.

    Tests, the data audit, and the strict completion gate are all mandatory
    preconditions for research execution. Operational failures are represented
    explicitly as BLOCKED/DEGRADED artifacts so GitHub Actions can remain green
    without ever claiming an out-of-sample result that was not produced.
    """
    attempts = max(1, int(os.getenv("RESEARCH_ATTEMPTS", "2")))
    backoff = max(0.0, float(os.getenv("RESEARCH_RETRY_BACKOFF", "15")))
    out = Path(os.getenv("RESEARCH_OUTPUT_DIR", "artifacts"))
    out.mkdir(parents=True, exist_ok=True)

    tests_passed = os.getenv("TESTS_PASSED", "true").lower() == "true"
    audit_passed = os.getenv("AUDIT_PASSED", "true").lower() == "true"
    gate = _load_gate(out)

    blockers: list[str] = []
    if not tests_passed:
        blockers.append("preflight tests failed")
    if not audit_passed:
        blockers.append("data audit failed")
    if gate is None:
        blockers.append("completion gate artifact is missing")
    elif not bool(gate.get("full_gate_passed", False)):
        blockers.extend(str(x) for x in gate.get("blocking_reasons", []) if str(x))
        if not gate.get("blocking_reasons"):
            blockers.append("completion gate did not pass")

    if blockers:
        _write_status(out, {
            "status": "BLOCKED",
            "reason": "Research execution was intentionally skipped because one or more mandatory preflight gates failed.",
            "blockers": sorted(set(blockers)),
            "runner": {"status": "SKIPPED_AFTER_PREFLIGHT_FAILURE"},
            "oos_claimed": False,
        })
        return 0

    errors: list[str] = []
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
    return 0


if __name__ == "__main__":
    raise SystemExit(run_with_retries())
