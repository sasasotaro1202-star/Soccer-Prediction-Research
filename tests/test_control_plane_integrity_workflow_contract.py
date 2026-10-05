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


def test_controller_allowlists_only_the_maturity_workflow_as_control_plane_hardening() -> None:
    workflow = Path(".github/workflows/autonomous-github-controller.yml").read_text(
        encoding="utf-8"
    )
    assert "soccer-prospective-inplay-maturity.yml" in workflow


def test_failure_recovery_has_scheduled_catchup_for_missed_workflow_events() -> None:
    workflow = Path(".github/workflows/action-failure-recovery.yml").read_text(
        encoding="utf-8"
    )
    assert "schedule:" in workflow
    assert 'cron: "*/15 * * * *"' in workflow
    assert "gh run list" in workflow
    assert "run_attempt" in workflow
    assert "gh run rerun" in workflow


def test_failure_recovery_has_no_escaped_github_expressions() -> None:
    workflow = Path(".github/workflows/action-failure-recovery.yml").read_text(
        encoding="utf-8"
    )
    assert r"\\${{" not in workflow
    assert "${{ github.event_name == 'schedule' }}" in workflow
    assert "${{ github.token }}" in workflow
    assert "${{ github.repository }}" in workflow


def test_controller_allowlists_world_model_supervisor_as_control_plane_hardening() -> None:
    workflow = Path(".github/workflows/autonomous-github-controller.yml").read_text(
        encoding="utf-8"
    )
    assert "soccer-world-model-supervisor.yml" in workflow


def test_soccer_ci_has_a_bounded_diagnostic_pytest_execution() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "timeout --signal=TERM --kill-after=30s 300" in workflow
    assert "python -m pytest -q -vv --durations=20" in workflow


def test_world_model_supervisor_parses_multi_document_decision_evidence() -> None:
    workflow = Path(".github/workflows/soccer-world-model-supervisor.yml").read_text(
        encoding="utf-8"
    )
    assert "JSONDecoder" in workflow
    assert "raw_decode" in workflow


def test_soccer_ci_ignores_machine_generated_world_model_evidence_pushes() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    required_paths = (
        '"data/research/prospective_inplay/raw/**"',
        '"data/research/match_state_snapshots/**"',
        '"data/research/match_state_research_status.json"',
    )
    for path in required_paths:
        assert path in workflow


def test_failure_recovery_passes_actions_history_through_a_file_not_argv() -> None:
    workflow = Path(".github/workflows/action-failure-recovery.yml").read_text(
        encoding="utf-8"
    )
    assert 'all_runs="$(gh api "repos/${GH_REPO}/actions/runs?per_page=100")"' not in workflow
    assert 'gh api "repos/${GH_REPO}/actions/runs?per_page=100" > /tmp/all-runs.json' in workflow
    assert 'Path(sys.argv[1]).read_text()' in workflow


def test_controller_rebases_safe_hardening_branches_after_main_moves() -> None:
    workflow = Path(".github/workflows/autonomous-github-controller.yml").read_text(
        encoding="utf-8"
    )
    assert "gh pr update-branch" in workflow
    assert "--rebase" in workflow
    assert "HOLD" in workflow


def test_failure_recovery_has_no_escaped_shell_variables() -> None:
    workflow = Path(".github/workflows/action-failure-recovery.yml").read_text(
        encoding="utf-8"
    )
    assert "\\${" not in workflow
    assert 'main_sha="$(gh api "repos/\\${GH_REPO}/git/ref/heads/main"' not in workflow
    assert 'for workflow in "\\${workflows[@]}"' not in workflow


def test_failure_recovery_emits_real_tab_delimiters() -> None:
    workflow = Path(".github/workflows/action-failure-recovery.yml").read_text(
        encoding="utf-8"
    )
    assert 'print("\\t".join([' in workflow
    assert 'print("\\\\t".join([' not in workflow


def test_maturity_lane_pr_validation_cannot_write_main() -> None:
    workflow = Path(".github/workflows/soccer-prospective-inplay-maturity.yml").read_text(
        encoding="utf-8"
    )
    assert "pull_request:" in workflow
    assert "github.event_name != 'pull_request'" in workflow
    assert "git push origin HEAD:main" in workflow


def test_failure_recovery_declares_top_level_schedule_trigger() -> None:
    workflow = Path(".github/workflows/action-failure-recovery.yml").read_text(
        encoding="utf-8"
    )
    assert '\n  schedule:\n    - cron: "*/15 * * * *"\n' in workflow


def test_failure_recovery_has_real_scheduled_trigger_not_just_a_job_name() -> None:
    workflow = Path(".github/workflows/action-failure-recovery.yml").read_text(
        encoding="utf-8"
    )
    assert 'on:\n  schedule:\n    - cron: "*/15 * * * *"\n  workflow_run:' in workflow


def test_maturity_pr_validation_checks_exact_pr_head() -> None:
    workflow = Path(".github/workflows/soccer-prospective-inplay-maturity.yml").read_text(
        encoding="utf-8"
    )
    assert "github.event_name == 'pull_request'" in workflow
    assert "ref: ${{ github.event_name == 'pull_request' && github.event.pull_request.head.sha || 'main' }}" in workflow


def test_failure_recovery_catchup_window_survives_extended_event_delivery_delays() -> None:
    workflow = Path(".github/workflows/action-failure-recovery.yml").read_text(
        encoding="utf-8"
    )
    assert "max_age_seconds=21600" in workflow


def test_failure_recovery_paginates_actions_history_for_extended_window() -> None:
    workflow = Path(".github/workflows/action-failure-recovery.yml").read_text(
        encoding="utf-8"
    )
    assert "per_page=100&page=" in workflow
    assert "oldest" in workflow
    assert "max_age_seconds" in workflow
