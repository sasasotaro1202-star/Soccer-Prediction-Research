from pathlib import Path


def test_integrity_audit_uses_timezone_aware_liveness_calculation() -> None:
    workflow = Path(".github/workflows/soccer-automation-integrity-audit.yml").read_text(
        encoding="utf-8"
    )
    assert "from datetime import datetime, timezone" in workflow
    assert 'datetime.fromisoformat(raw.replace("Z", "+00:00"))' in workflow
    assert "created_at.astimezone(timezone.utc).timestamp()" in workflow
    assert 'time.mktime(time.strptime(raw, "%Y-%m-%dT%H:%M:%SZ"))' not in workflow


def test_integrity_audit_remains_read_only_and_fail_closed() -> None:
    workflow = Path(".github/workflows/soccer-automation-integrity-audit.yml").read_text(
        encoding="utf-8"
    )
    assert "permissions:" in workflow
    assert "actions: read" in workflow
    assert "contents: read" in workflow
    assert "CONTROL-PLANE INTEGRITY AUDIT HOLD/FAIL" in workflow
    assert '"production_change_allowed": False' in workflow
    assert '"performance_claim_allowed": False' in workflow
    assert '"frozen_holdout_access_allowed": False' in workflow


def test_integrity_audit_covers_all_continuous_world_model_lanes() -> None:
    workflow = Path(".github/workflows/soccer-automation-integrity-audit.yml").read_text(
        encoding="utf-8"
    )
    required_lanes = (
        "Soccer Prospective In-Play PIT Capture",
        "Soccer Prospective In-Play Maturity",
        "Soccer World Model Supervisor",
        "Soccer Dynamic Simulator Research",
        "Soccer Match State Research",
    )
    for lane in required_lanes:
        assert lane in workflow

    required_schedules = (
        'soccer-prospective-inplay-capture.yml',
        'soccer-prospective-inplay-maturity.yml',
        'soccer-world-model-supervisor.yml',
        'soccer-dynamic-simulator-research.yml',
    )
    for workflow_file in required_schedules:
        assert workflow_file in workflow


def test_integrity_audit_does_not_treat_recent_failed_runs_as_healthy() -> None:
    workflow = Path(".github/workflows/soccer-automation-integrity-audit.yml").read_text(
        encoding="utf-8"
    )
    assert '"FAILED_RECENT"' in workflow
    assert '"OK_ACTIVE"' in workflow
    assert 'latest.get("conclusion") != "success"' in workflow
    assert "status_raw" in workflow
    assert "conclusion" in workflow


def test_integrity_audit_checks_world_model_liveness_windows() -> None:
    workflow = Path(".github/workflows/soccer-automation-integrity-audit.yml").read_text(
        encoding="utf-8"
    )
    expected_windows = (
        '"soccer-prospective-inplay-capture.yml": 45 * 60',
        '"soccer-prospective-inplay-maturity.yml": 90 * 60',
        '"soccer-world-model-supervisor.yml": 5 * 3600',
        '"soccer-dynamic-simulator-research.yml": 12 * 3600',
        '"soccer-match-state-research.yml": 12 * 3600',
    )
    for marker in expected_windows:
        assert marker in workflow


def test_integrity_audit_covers_recovery_for_each_world_model_lane() -> None:
    workflow = Path(".github/workflows/soccer-automation-integrity-audit.yml").read_text(
        encoding="utf-8"
    )
    recovery_markers = (
        "Soccer Prospective In-Play PIT Capture",
        "Soccer Prospective In-Play Maturity",
        "Soccer World Model Supervisor",
        "Soccer Dynamic Simulator Research",
        "Soccer Match State Research",
    )
    for marker in recovery_markers:
        assert marker in workflow


def test_action_failure_recovery_lists_all_world_model_lanes() -> None:
    workflow = Path(".github/workflows/action-failure-recovery.yml").read_text(
        encoding="utf-8"
    )
    required_lanes = (
        "Soccer Prospective In-Play PIT Capture",
        "Soccer Prospective In-Play Maturity",
        "Soccer World Model Supervisor",
        "Soccer Dynamic Simulator Research",
        "Soccer Match State Research",
    )
    for lane in required_lanes:
        assert lane in workflow


def test_controller_reconciles_after_integrity_audit_completion() -> None:
    workflow = Path(".github/workflows/autonomous-github-controller.yml").read_text(
        encoding="utf-8"
    )
    assert "Soccer Automation Integrity Audit" in workflow
    assert "types: [completed]" in workflow


def test_world_model_supervisor_does_not_hide_dispatch_failures() -> None:
    workflow = Path(".github/workflows/soccer-world-model-supervisor.yml").read_text(
        encoding="utf-8"
    )
    assert "dispatch_failures=0" in workflow
    assert "dispatch_failures=$((dispatch_failures + 1))" in workflow
    assert 'if [ "$dispatch_failures" -gt 0 ]; then' in workflow
    assert "exit 1" in workflow


def test_world_model_maturity_installs_python_dependencies_before_pytest() -> None:
    workflow = Path(".github/workflows/soccer-prospective-inplay-maturity.yml").read_text(
        encoding="utf-8"
    )
    assert "actions/setup-python@" in workflow
    assert 'python-version: "3.12"' in workflow
    assert "python -m pip install" in workflow
    assert "requirements.txt" in workflow
