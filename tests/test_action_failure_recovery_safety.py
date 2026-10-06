from scripts.action_failure_recovery import classify_recovery


def test_deterministic_failed_test_step_is_not_retryable():
    jobs = {
        "jobs": [
            {
                "steps": [
                    {"name": "Run deterministic test suite", "conclusion": "failure"},
                ]
            }
        ]
    }
    decision = classify_recovery(conclusion="failure", run_attempt=1, jobs_payload=jobs)
    assert decision["retryable"] is False
    assert decision["reason"] == "deterministic_or_unknown_failure"
    assert decision["blocked_steps"] == ["Run deterministic test suite"]


def test_transient_setup_failure_is_retryable_once():
    jobs = {
        "jobs": [
            {
                "steps": [
                    {"name": "Install dependencies with bounded retry", "conclusion": "failure"},
                ]
            }
        ]
    }
    decision = classify_recovery(conclusion="failure", run_attempt=1, jobs_payload=jobs)
    assert decision["retryable"] is True
    assert decision["reason"] == "transient_setup_failure"


def test_timeout_is_retryable_only_on_first_attempt():
    first = classify_recovery(conclusion="timed_out", run_attempt=1, jobs_payload={"jobs": []})
    second = classify_recovery(conclusion="timed_out", run_attempt=2, jobs_payload={"jobs": []})
    assert first["retryable"] is True
    assert second["retryable"] is False


def test_mixed_transient_and_deterministic_failures_are_blocked():
    jobs = {
        "jobs": [
            {
                "steps": [
                    {"name": "Checkout", "conclusion": "failure"},
                    {"name": "Run tests", "conclusion": "failure"},
                ]
            }
        ]
    }
    decision = classify_recovery(conclusion="failure", run_attempt=1, jobs_payload=jobs)
    assert decision["retryable"] is False
    assert decision["blocked_steps"] == ["Run tests"]


def test_recovery_workflow_requires_classifier_before_rerun():
    from pathlib import Path

    text = Path(".github/workflows/action-failure-recovery.yml").read_text(encoding="utf-8")
    classify_pos = text.index("action_failure_recovery.py")
    rerun_pos = text.index("gh run rerun")
    assert classify_pos < rerun_pos
    assert "retryable" in text
    assert "deterministic/unknown failure remains blocking" in text


def test_incomplete_job_details_are_not_retryable():
    jobs = {
        "total_count": 101,
        "jobs": [
            {"steps": [{"name": "Checkout", "conclusion": "failure"}]}
        ],
    }
    decision = classify_recovery(conclusion="failure", run_attempt=1, jobs_payload=jobs)
    assert decision["retryable"] is False
    assert decision["reason"] == "job_detail_incomplete"


def test_empty_job_details_are_not_retryable():
    decision = classify_recovery(conclusion="failure", run_attempt=1, jobs_payload={"jobs": []})
    assert decision["retryable"] is False
    assert decision["reason"] == "job_detail_empty_or_invalid"
