from pathlib import Path


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
    assert 'cron: "*/10 * * * *"' in text
    assert "group: action-failure-recovery" in text
    assert "Soccer Prospective In-Play PIT Capture" in text
    assert "Soccer Prospective In-Play Maturity" in text
    assert "Soccer World Model Supervisor" in text
    assert "Soccer Match State Research" in text
    assert "Soccer Daily Research Forecast" in text
    assert "Soccer Live Research Forecast" in text
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
    assert "GITHUB_EVENT_NAME" in text
    assert "pull_request" in text
    assert "PR validation only; main persistence disabled." in text


def test_ci_bounds_tests_and_ignores_generated_world_model_data() -> None:
    text = _read(".github/workflows/ci.yml")
    assert "Validate workflow YAML syntax" in text
    assert "timeout --signal=TERM --kill-after=30s 3600" in text
    assert "python -m pytest -q -vv --durations=20" in text
    assert "data/research/prospective_inplay/raw/**" in text
    assert "data/research/match_state_snapshots/**" in text


def test_controller_self_advances_safe_hardening_prs() -> None:
    text = _read(".github/workflows/autonomous-github-controller.yml")
    assert "pull_request:" in text
    assert "gh pr update-branch" in text
    assert "--rebase" in text
    assert "hardening-safe-v1" in text
    assert "--match-head-commit" in text
    assert '[ "${head_repo}" = "${GH_REPO}" ]' in text


def test_control_plane_event_graph_has_no_audit_recovery_controller_cycle() -> None:
    audit = _read(".github/workflows/soccer-automation-integrity-audit.yml")
    recovery = _read(".github/workflows/action-failure-recovery.yml")
    controller = _read(".github/workflows/autonomous-github-controller.yml")
    audit_triggers = audit.split("permissions:", 1)[0]
    assert "workflow_run:" not in audit_triggers
    assert "workflow_run:" not in recovery
    assert "workflow_run:" not in controller

    required_control_filenames = (
        "soccer-autonomous-orchestrator.yml",
        "soccer-control-plane-watchdog.yml",
        "action-failure-recovery.yml",
        "autonomous-github-controller.yml",
        "soccer-9h-queue-watchdog.yml",
        "soccer-9h-recovery.yml",
        "soccer-automation-heartbeat.yml",
    )
    for name in required_control_filenames:
        assert "'" + name + "'" in audit
        assert name in audit

    for isolated in (
        ".github/workflows/soccer-9h-recovery.yml",
        ".github/workflows/soccer-world-model-supervisor.yml",
        ".github/workflows/soccer-matchday-intelligence.yml",
    ):
        assert "workflow_run:" not in _read(isolated)


def test_prospective_capture_collects_three_spaced_observations():
    text = _read(".github/workflows/soccer-prospective-inplay-capture.yml")
    assert "--loops 3" in text
    assert "--interval-seconds 240" in text
    assert "No new prospective in-play snapshots." in text


def test_replayed_oos_selects_only_successful_robust_artifacts():
    text = _read(".github/workflows/soccer-replayed-oos-performance.yml")
    assert "workflow_run.get('status') != 'completed'" in text
    assert "workflow_run.get('conclusion') != 'success'" in text
    assert "Workflow-run replay requires a successful Soccer Research Robust completion." in text


def test_legacy_bridge_is_source_scoped():
    text = _read(".github/workflows/legacy-bridge-check.yml")
    assert "paths:" in text
    assert "paths-ignore:" not in text
    assert "src/research/**" in text
    assert "tests/test_baseline_comparison.py" in text
    assert "data/experience/**" not in text
    assert "artifacts/**" not in text


def test_legacy_research_cycle_is_not_orchestrated() -> None:
    orchestrator = _read(".github/workflows/soccer-autonomous-orchestrator.yml")
    assert 'ensure_lane "soccer-research-cycle.yml"' not in orchestrator
