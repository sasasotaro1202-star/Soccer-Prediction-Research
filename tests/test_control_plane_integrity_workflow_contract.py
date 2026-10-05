from pathlib import Path
import re

def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def test_integrity_audit_is_read_only_and_fail_closed() -> None:
    text = _read(".github/workflows/soccer-automation-integrity-audit.yml")
    assert "actions: read" in text
    assert "contents: read" in text
    assert "'production_change_allowed': False" in text
    assert "'frozen_holdout_access_allowed': False" in text
    assert "'performance_claim_allowed': False" in text
    assert "AUTOMATION_INTEGRITY: PASS" in text

def test_failure_recovery_is_scheduled_and_covers_world_model() -> None:
    text = _read(".github/workflows/action-failure-recovery.yml")
    assert 'cron: "*/15 * * * *"' in text
    assert "group: action-failure-recovery" in text
    assert "Soccer Prospective In-Play PIT Capture" in text
    assert "Soccer Prospective In-Play Maturity" in text
    assert "Soccer World Model Supervisor" in text
    assert "Soccer Match State Research" in text
    assert "max_age_seconds=21600" in text
    assert "run_attempt" in text
    assert "exit 1" in text

def test_world_model_supervisor_fails_closed_on_dispatch_failure() -> None:
    text = _read(".github/workflows/soccer-world-model-supervisor.yml")
    assert "dispatch_failures=0" in text
    assert "dispatch_failures=$((dispatch_failures + 1))" in text
    assert 'if [ "$dispatch_failures" -gt 0 ]; then' in text
    assert "world_model_dispatch_failures" in text
    assert "JSONDecoder" in text
    assert "raw_decode" in text

def test_maturity_bootstraps_dependencies_and_is_pr_safe() -> None:
    text = _read(".github/workflows/soccer-prospective-inplay-maturity.yml")
    assert "actions/setup-python@" in text
    assert "python -m pip install" in text
    assert "github.event_name == 'pull_request' && github.event.pull_request.head.sha || 'main'" in text
    assert "github.event_name != 'pull_request'" in text

def test_ci_bounds_tests_and_ignores_generated_world_model_data() -> None:
    text = _read(".github/workflows/ci.yml")
    assert "Validate workflow YAML syntax" in text
    assert "timeout-minutes: 30" in text
    assert "timeout --signal=TERM --kill-after=60s 900" in text
    assert "python -m pytest -q -vv --durations=20" in text
    assert "data/research/prospective_inplay/raw/**" in text
    assert "data/research/match_state_snapshots/**" in text

def test_controller_self_advances_safe_hardening_prs() -> None:
    text = _read(".github/workflows/autonomous-github-controller.yml")
    assert "gh pr update-branch" in text
    assert "--rebase" in text
    assert "hardening-safe-v1" in text
    assert "--match-head-commit" in text
    assert "[ \"${head_repo}\" = \"${GH_REPO}\" ]" in text

def test_control_plane_event_graph_has_no_audit_recovery_controller_cycle() -> None:
    audit = _read(".github/workflows/soccer-automation-integrity-audit.yml")
    recovery = _read(".github/workflows/action-failure-recovery.yml")
    controller = _read(".github/workflows/autonomous-github-controller.yml")
    assert "Action Failure Recovery" not in audit
    assert "Autonomous GitHub Controller" not in audit
    workflow_run_block = re.search(
        r"workflow_run:\n(.*?)(?=\n\s*types:)",
        recovery,
        re.DOTALL,
    )
    assert workflow_run_block is not None
    assert "Soccer Automation Integrity Audit" not in workflow_run_block.group(1)
    assert "Autonomous GitHub Controller" not in workflow_run_block.group(1)
    assert "Soccer Automation Integrity Audit" not in controller
    assert "Soccer Autonomous Orchestrator" in audit
    assert "Soccer Control Plane Watchdog" in audit