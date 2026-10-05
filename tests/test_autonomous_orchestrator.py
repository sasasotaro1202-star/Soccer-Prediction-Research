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
    assert 'r.get("conclusion") == "success"' in text
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
    assert "RUNNING_STALE" in text
    assert "QUEUED_WAIT" in text
    assert "DISPATCH" in text
    assert "ACTIVE" in text
    assert "max-time 90" in text
    assert "for attempt in 1 2 3 4" in text


def test_control_plane_watchdog_is_independent_and_persists_evidence():
    path = Path(".github/workflows/soccer-control-plane-watchdog.yml")
    text = path.read_text(encoding="utf-8")
    assert "workflow_run:" not in text
    assert 'cron: "*/10 * * * *"' in text
    assert "export DECISION" in text
    assert "upload-artifact@" in text
    assert "control_plane_watchdog.json" in text


def test_failure_recovery_is_schedule_only_and_bounded():
    path = Path(".github/workflows/action-failure-recovery.yml")
    text = path.read_text(encoding="utf-8")
    assert "workflow_run:" not in text
    assert 'cron: "*/10 * * * *"' in text
    assert "Soccer Control Plane Watchdog" not in text
    assert "Soccer Automation Heartbeat" not in text
    assert "Soccer Automation Integrity Audit" not in text
    assert "timed_out" in text
    assert "rerun_budget=2" in text


def test_autonomous_controller_is_not_a_workflow_run_event_sink():
    path = Path(".github/workflows/autonomous-github-controller.yml")
    text = path.read_text(encoding="utf-8")
    assert "group: soccer-autonomous-controller" in text
    assert "cancel-in-progress: true" in text
    assert "workflow_run:" not in text
    assert "pull_request:" in text
    assert 'cron: "17 * * * *"' in text


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


def test_failure_recovery_excludes_non_source_control_plane_workflows():
    text = Path(".github/workflows/action-failure-recovery.yml").read_text(encoding="utf-8")
    assert "Soccer Automation Heartbeat" not in text
    assert "Soccer Automation Integrity Audit" not in text


def test_failure_recovery_isolated_per_workflow_lane():
    text = Path(".github/workflows/action-failure-recovery.yml").read_text(encoding="utf-8")
    assert "group: action-failure-recovery" in text
    assert "cancel-in-progress: true" in text
    assert 'cron: "*/10 * * * *"' in text


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
    assert "gh pr merge" not in audit
    assert "control_plane_masking=0" in audit
    assert "Workflow is not active in GitHub Actions" in audit
    assert "soccer-automation-integrity-audit.yml" in orchestrator
    assert "Soccer Automation Integrity Audit" in recovery


def test_control_plane_watchdog_does_not_mask_errors_with_or_true():
    text = Path(".github/workflows/soccer-control-plane-watchdog.yml").read_text(encoding="utf-8")
    assert "|| true" not in text


def test_orchestrator_reaps_only_obsolete_prestart_runs():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "cancel_obsolete_prestart_runs" in text
    assert '.status == "queued" or .status == "pending" or .status == "waiting" or .status == "requested"' in text
    assert '.headSha != $main_sha' in text
    assert 'gh run cancel "${run_id}" --repo "${GH_REPO}"' in text
    assert "obsolete_prestart_reap_budget_exhausted" in text


def test_orchestrator_reacts_to_control_plane_changes():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'push:' in text
    assert 'branches: [main]' in text
    assert '".github/workflows/**"' in text
    assert '"PROJECT_INSTRUCTIONS.md"' in text
    assert '"docs/PROJECT_SOURCE.md"' in text

 
 
def test_automation_integrity_rejects_controller_event_sink():
    audit = Path(".github/workflows/soccer-automation-integrity-audit.yml").read_text(encoding="utf-8")
    assert "must not subscribe to workflow_run events" in audit

 
 
def test_action_recovery_excludes_control_plane_event_sources():
    text = Path(".github/workflows/action-failure-recovery.yml").read_text(encoding="utf-8")
    assert "Soccer Control Plane Watchdog" not in text
    assert "Soccer Automation Heartbeat" not in text
    assert "Soccer Automation Integrity Audit" not in text



def test_completion_event_isolation_for_state_supervisors():
    for relative in (
        ".github/workflows/soccer-9h-recovery.yml",
        ".github/workflows/soccer-world-model-supervisor.yml",
        ".github/workflows/soccer-matchday-intelligence.yml",
    ):
        text = Path(relative).read_text(encoding="utf-8")
        assert "workflow_run:" not in text
