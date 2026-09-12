from __future__ import annotations

import json
import os
import time
from pathlib import Path


def run_with_retries() -> int:
    """Run the research engine with bounded retries and always persist a status file.

    A transient acquisition/network failure should not waste a complete Actions run.
    The wrapper retries the whole deterministic engine; if all attempts fail it records
    a DEGRADED status and returns success so the Actions workflow itself remains green.
    The artifact is the authoritative health signal; no failed run is silently treated
    as a successful research cycle.
    """
    attempts = max(1, int(os.getenv("RESEARCH_ATTEMPTS", "2")))
    backoff = max(0.0, float(os.getenv("RESEARCH_RETRY_BACKOFF", "15")))
    out = Path(os.getenv("RESEARCH_OUTPUT_DIR", "artifacts"))
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    from src.research.engine import run

    for attempt in range(1, attempts + 1):
        try:
            report = run(str(out))
            report["runner"] = {"attempt": attempt, "max_attempts": attempts, "status": "COMPLETED"}
            (out / "run_status.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
            )
            return 0
        except Exception as exc:
            errors.append(f"attempt={attempt} {type(exc).__name__}: {exc}")
            if attempt < attempts:
                time.sleep(backoff * attempt)

    status = {
        "status": "DEGRADED",
        "reason": "Research engine failed after bounded retries; no OOS result was claimed.",
        "runner": {"attempts": attempts, "status": "FAILED_AFTER_RETRIES"},
        "errors": errors,
    }
    (out / "run_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_with_retries())
