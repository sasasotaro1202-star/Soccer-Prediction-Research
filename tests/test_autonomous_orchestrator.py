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


def test_orchestrator_has_bounded_dispatch_and_active_run_caps():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "dispatch_budget=6" in text
    assert "max_active_runs=10" in text
    assert "dispatch_budget_exhausted" in text
    assert "active_run_cap" in text


def test_orchestrator_does_not_stop_on_single_dispatch_error():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'if case "${workflow}" in' in text
    assert '"reason":"dispatch_error"' in text
    assert "continue reconciliation" in text


def test_orchestrator_retries_dispatch_and_isolates_state_query_errors():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "dispatch_workflow()" in text
    assert "for attempt in 1 2 3" in text
    assert '"reason":"workflow_state_error"' in text
    assert '"reason":"active_run_count_error"' in text


def test_orchestrator_caps_current_main_runs_without_legacy_blocking():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "active_current_main_count()" in text
    assert "headSha == $main_sha" in text
    assert "active_current_main_cap" in text
    assert "max_active_current_main_runs: 10" in text
