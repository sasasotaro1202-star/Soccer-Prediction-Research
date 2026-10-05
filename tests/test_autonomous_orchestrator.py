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
