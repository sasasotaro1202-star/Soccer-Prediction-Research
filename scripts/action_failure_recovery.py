from __future__ import annotations

import json
import sys
from typing import Any

# Only these setup/network steps are safe to retry automatically after a normal
# workflow failure. Unknown failed steps remain blocking so deterministic test,
# schema, PIT, OOS, production, or evidence failures cannot be hidden by reruns.
TRANSIENT_FAILURE_STEPS = frozenset(
    {
        "Set up job",
        "Checkout",
        "Set up Python",
        "Install dependencies with bounded retry",
    }
)


def classify_recovery(*, conclusion: str, run_attempt: int, jobs_payload: dict[str, Any]) -> dict[str, Any]:
    if run_attempt != 1:
        return {
            "retryable": False,
            "reason": "run_attempt_is_not_first",
            "failed_steps": [],
            "blocked_steps": [],
        }

    if conclusion == "timed_out":
        return {
            "retryable": True,
            "reason": "workflow_timed_out",
            "failed_steps": [],
            "blocked_steps": [],
        }

    if conclusion != "failure":
        return {
            "retryable": False,
            "reason": f"unsupported_conclusion:{conclusion}",
            "failed_steps": [],
            "blocked_steps": [],
        }

    jobs = jobs_payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        return {
            "retryable": False,
            "reason": "job_detail_empty_or_invalid",
            "failed_steps": [],
            "blocked_steps": [],
        }

    total_count = jobs_payload.get("total_count")
    if isinstance(total_count, int) and total_count > len(jobs):
        return {
            "retryable": False,
            "reason": "job_detail_incomplete",
            "failed_steps": [],
            "blocked_steps": [],
        }

    failed_steps: list[str] = []
    blocked_steps: list[str] = []
    for job in jobs:
        if not isinstance(job, dict):
            return {
                "retryable": False,
                "reason": "job_detail_invalid",
                "failed_steps": sorted(set(failed_steps)),
                "blocked_steps": sorted(set(blocked_steps)),
            }
        steps = job.get("steps", [])
        if steps is None:
            steps = []
        if not isinstance(steps, list):
            return {
                "retryable": False,
                "reason": "job_steps_invalid",
                "failed_steps": sorted(set(failed_steps)),
                "blocked_steps": sorted(set(blocked_steps)),
            }
        for step in steps:
            if not isinstance(step, dict):
                return {
                    "retryable": False,
                    "reason": "job_step_detail_invalid",
                    "failed_steps": sorted(set(failed_steps)),
                    "blocked_steps": sorted(set(blocked_steps)),
                }
            if step.get("conclusion") != "failure":
                continue
            name = str(step.get("name") or "<unnamed failed step>")
            failed_steps.append(name)
            if name not in TRANSIENT_FAILURE_STEPS:
                blocked_steps.append(name)

    if not failed_steps:
        return {
            "retryable": False,
            "reason": "job_detail_has_no_failed_step",
            "failed_steps": [],
            "blocked_steps": [],
        }

    retryable = not blocked_steps
    reason = "transient_setup_failure" if retryable else "deterministic_or_unknown_failure"
    return {
        "retryable": retryable,
        "reason": reason,
        "failed_steps": sorted(set(failed_steps)),
        "blocked_steps": sorted(set(blocked_steps)),
    }


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: action_failure_recovery.py <conclusion> <run_attempt>")
    conclusion = sys.argv[1]
    try:
        run_attempt = int(sys.argv[2])
    except ValueError as exc:
        raise SystemExit("run_attempt must be an integer") from exc

    payload = json.load(sys.stdin)
    decision = classify_recovery(
        conclusion=conclusion,
        run_attempt=run_attempt,
        jobs_payload=payload,
    )
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
