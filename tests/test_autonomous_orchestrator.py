from pathlib import Path


WORKFLOW = Path(".github/workflows/soccer-autonomous-orchestrator.yml")


def test_orchestrator_is_fail_closed_and_main_pinned():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "permissions:" in text
    assert "actions: write" in text
    assert "contents: read" in text
    assert 'git/ref/heads/main' in text
    assert 'main_sha="$(api' in text
    assert 'gh workflow run "${workflow}" --repo "${GH_REPO}" --ref main' in text
    assert 'r.get("headSha") == main_sha' in text
    assert 'latest_conclusion == "success"' in text
    assert "DISPATCH:no_current_main_history" in text
    assert '"production_change_allowed": False' in text
    assert '"frozen_holdout_access_allowed": False' in text
    assert '"performance_claim_allowed": False' in text


def test_orchestrator_has_pit_and_research_reconciliation_lanes():
    text = WORKFLOW.read_text(encoding="utf-8")
    required = (
        "soccer-9h-autonomous.yml",
        "soccer-research-robust.yml",
        "soccer-predictability-research.yml",
        "soccer-historical-learning.yml",
        "soccer-experience-ledger.yml",
        "adaptive_data_discovery.yml",
        "soccer-scope-frontier.yml",
        "pit-replay-audit.yml",
        "soccer-coverage-pit-audit.yml",
        "soccer-audit.yml",
        "soccer-versioned-pit-research.yml",
        "soccer-daily-research-forecast.yml",
        "soccer-matchday-intelligence.yml",
        "soccer-research-cycle.yml",
        "overnight-integrity.yml",
        "soccer-opta-like-24h.yml",
    )
    for workflow in required:
        assert workflow in text


def test_orchestrator_does_not_allow_performance_or_holdout_mutation():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "frozen_holdout_access_allowed" in text
    assert "performance_claim_allowed" in text
    assert "production_change_allowed" in text
    assert "gh pr merge" not in text




def test_main_sha_export_precedes_report_initialization():
    text = WORKFLOW.read_text(encoding="utf-8")
    export_pos = text.index('export MAIN_SHA="${main_sha}"')
    report_pos = text.index("python - <<'PY' > \"$report\"")
    assert export_pos < report_pos


def test_orchestrator_supplies_required_pit_replay_inputs():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "pit-replay-audit.yml)" in text
    assert "-f competition=EPL" in text
    assert "-f rows_per_season=1" in text


def test_control_plane_watchdog_has_catchup_and_restart_guards():
    path = Path(".github/workflows/soccer-control-plane-watchdog.yml")
    text = path.read_text(encoding="utf-8")
    assert 'cron: "*/10 * * * *"' in text
    assert "RESTART" in text
    assert "DISPATCH" in text
    assert "ACTIVE" in text
    assert "max-time 90" in text
    assert "for attempt in 1 2 3 4" in text


def test_control_plane_watchdog_is_event_driven_and_persists_evidence():
    path = Path(".github/workflows/soccer-control-plane-watchdog.yml")
    text = path.read_text(encoding="utf-8")
    assert "workflow_run:" in text
    assert "Soccer Autonomous Orchestrator" in text
    assert "Soccer Research Robust" in text
    assert "Soccer 9H Autonomous Research" in text
    assert "Soccer Research Cycle" in text
    assert "Soccer PIT Replay Audit" in text
    assert "export DECISION" in text
    assert "upload-artifact@" in text
    assert "control_plane_watchdog.json" in text


def test_failure_recovery_includes_watchdog_timeout():
    path = Path(".github/workflows/action-failure-recovery.yml")
    text = path.read_text(encoding="utf-8")
    assert "Soccer Control Plane Watchdog" in text
    assert "timed_out" in text


def test_autonomous_controller_cancels_stale_reconciliations():
    path = Path(".github/workflows/autonomous-github-controller.yml")
    text = path.read_text(encoding="utf-8")
    assert "group: soccer-autonomous-controller" in text
    assert "cancel-in-progress: true" in text
    assert ".github/workflows/soccer-control-plane-watchdog.yml" in text


def test_automation_heartbeat_is_long_term_and_bounded():
    path = Path(".github/workflows/soccer-automation-heartbeat.yml")
    text = path.read_text(encoding="utf-8")
    assert 'cron: "17 4 * * 0"' in text
    assert "contents: write" in text
    assert "chore: automation heartbeat" in text
    assert "for attempt in 1 2 3 4" in text
    assert "--max-time 90" in text
    assert "production_change_allowed" in text
    assert "frozen_holdout_access_allowed" in text


def test_heartbeat_path_is_ignored_by_heavy_push_checks():
    ci = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    legacy = Path(".github/workflows/legacy-bridge-check.yml").read_text(encoding="utf-8")
    assert ".github/automation/**" in ci
    assert ".github/automation/**" in legacy


def test_failure_recovery_includes_automation_heartbeat():
    text = Path(".github/workflows/action-failure-recovery.yml").read_text(encoding="utf-8")
    assert "Soccer Automation Heartbeat" in text


def test_failure_recovery_isolated_per_workflow_lane():
    text = Path(".github/workflows/action-failure-recovery.yml").read_text(encoding="utf-8")
    assert "group: action-failure-recovery-${{ github.event.workflow_run.name }}" in text
    assert "cancel-in-progress: true" in text


def test_orchestrator_backs_off_repeated_failures():
    text = Path(".github/workflows/soccer-autonomous-orchestrator.yml").read_text(encoding="utf-8")
    assert "failure_backoff" in text
    assert "retry_after_failure_backoff" in text
    assert "24 * 3600" in text


def test_orchestrator_includes_continuous_world_model_lanes():
    text = WORKFLOW.read_text(encoding="utf-8")
    required = (
        "soccer-prospective-inplay-capture.yml",
        "soccer-prospective-inplay-maturity.yml",
        "soccer-world-model-supervisor.yml",
    )
    for workflow in required:
        assert workflow in text


def test_orchestrator_keeps_world_model_research_non_authoritative():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "production_change_allowed" in text
    assert "frozen_holdout_access_allowed" in text
    assert "performance_claim_allowed" in text
    assert "gh pr merge" not in text


def test_automation_integrity_audit_is_required_and_fail_closed():
    audit = Path(".github/workflows/soccer-automation-integrity-audit.yml").read_text(encoding="utf-8")
    orchestrator = WORKFLOW.read_text(encoding="utf-8")
    recovery = Path(".github/workflows/action-failure-recovery.yml").read_text(encoding="utf-8")
    assert 'cron: "13 */6 * * *"' in audit
    assert "workflow_dispatch:" in audit
    assert "concurrency:" in audit
    assert "actions: read" in audit
    assert "contents: read" in audit
    assert "ensure_lane" in audit
    assert "if 'gh pr merge' in orchestrator:" in audit
    assert "control_plane_masking=0" in audit
    assert "Workflow is not active in GitHub Actions" in audit
    assert "soccer-automation-integrity-audit.yml" in orchestrator
    assert "Soccer Automation Integrity Audit" in recovery


def test_control_plane_watchdog_does_not_mask_errors_with_or_true():
    text = Path(".github/workflows/soccer-control-plane-watchdog.yml").read_text(encoding="utf-8")
    assert "|| true" not in text

def test_orchestrator_never_masks_newest_failure_with_older_success():
    text = WORKFLOW.read_text(encoding="utf-8")
    latest_pos = text.index("latest_completed = current_completed[0]")
    conclusion_pos = text.index("latest_conclusion = latest_completed.get")
    success_pos = text.index('if latest_conclusion == "success":')
    failure_pos = text.index('if latest_conclusion in {"failure", "timed_out", "cancelled"}:')
    assert latest_pos < conclusion_pos < success_pos < failure_pos
    assert "successful = [" not in text
    assert "HOLD:unknown_terminal:" in text

def test_orchestrator_can_self_heal_disabled_lanes_with_bounded_retries():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_activation_state" in text
    assert "ensure_workflow_active" in text
    assert 'gh workflow enable "${workflow}" --repo "${GH_REPO}"' in text
    assert "workflow_inactive_unrecoverable" in text
    assert text.count("for attempt in 1 2 3;") >= 2

def test_watchdog_can_reenable_core_control_plane_without_prediction_authority():
    text = Path(".github/workflows/soccer-control-plane-watchdog.yml").read_text(encoding="utf-8")
    required = (
        "soccer-autonomous-orchestrator.yml",
        "action-failure-recovery.yml",
        "autonomous-github-controller.yml",
        "soccer-automation-heartbeat.yml",
    )
    for workflow in required:
        assert workflow in text
    assert "ensure_core_control_plane_active" in text
    assert "actions/workflows/${workflow}/enable" in text
    assert "for attempt in 1 2 3; do" in text
    assert "production_change_allowed" in text
    assert "frozen_holdout_access_allowed" in text
    assert "performance_claim_allowed" in text

def test_research_maturity_skips_safe_hardening_prs():
    text = Path(".github/workflows/soccer-prospective-inplay-maturity.yml").read_text(encoding="utf-8")
    assert "github.event_name != 'pull_request'" in text
    assert "startsWith(github.event.pull_request.head.ref, 'hardening/')" in text
    assert "Research-only maturation is not a required gate for hardening-safe PRs." in text

def test_ci_keeps_deterministic_suite_bounded_but_allows_slow_regressions():
    text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "timeout-minutes: 30" in text
    assert "timeout --signal=TERM --kill-after=60s 900" in text
    assert "python -m pytest -q -vv --durations=20" in text


def test_heartbeat_can_reenable_core_control_plane_without_masking_errors():
    text = Path(".github/workflows/soccer-automation-heartbeat.yml").read_text(encoding="utf-8")
    assert "actions: write" in text
    assert "reenable_core_workflow()" in text
    assert "actions/workflows/${workflow}" in text
    assert "gh workflow enable \"${workflow}\" --repo \"${GH_REPO}\"" in text
    assert "for attempt in 1 2 3; do" in text
    assert "|| true" not in text
    assert "production_change_allowed" in text
    assert "frozen_holdout_access_allowed" in text


def test_autonomous_controller_allowlists_heartbeat_as_safe_control_plane():
    text = Path(".github/workflows/autonomous-github-controller.yml").read_text(encoding="utf-8")
    assert ".github/workflows/soccer-automation-heartbeat.yml" in text
    assert "production/model/data/artifact mutation paths" not in text


def test_autonomous_controller_allowlists_project_instruction_contract():
    text = Path(".github/workflows/autonomous-github-controller.yml").read_text(encoding="utf-8")
    assert "PROJECT_INSTRUCTIONS.md|tests/*|docs/*" in text
