from pathlib import Path

def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def test_integrity_audit_is_read_only_and_fail_closed() -> None:
    workflow = _read(".github/workflows/soccer-automation-integrity-audit.yml")
    assert "actions: read" in workflow
    assert "contents: read" in workflow
    assert '"production_change_allowed": False' in workflow
    assert '"performance_claim_allowed": False' in workflow
    assert '"frozen_holdout_access_allowed": False' in workflow
    assert "CONTROL-PLANE INTEGRITY AUDIT HOLD/FAIL" in workflow

def test_integrity_audit_covers_all_world_model_lanes() -> None:
    workflow = _read(".github/workflows/soccer-automation-integrity-audit.yml")
    for marker in (
        "Soccer Prospective In-Play PIT Capture",
        "Soccer Prospective In-Play Maturity",
        "Soccer World Model Supervisor",
        "Soccer Dynamic Simulator Research",
        "Soccer Match State Research",
    ):
        assert marker in workflow

def test_integrity_audit_uses_current_main_liveness_and_explicit_failure_states() -> None:
    workflow = _read(".github/workflows/soccer-automation-integrity-audit.yml")
    assert "git/ref/heads/main" in workflow
    assert "OK_ACTIVE" in workflow
    assert "FAILED_RECENT" in workflow
    assert "STALE" in workflow
    assert 'datetime.fromisoformat(raw.replace("Z", "+00:00"))' in workflow

def test_failure_recovery_trigger_and_concurrency_are_deterministic() -> None:
    workflow = _read(".github/workflows/action-failure-recovery.yml")
    assert 'cron: "*/15 * * * *"' in workflow
    assert "group: action-failure-recovery" in workflow
    assert "github.event.workflow_run.name" not in workflow
    assert "\n  push:" not in workflow
    assert "workflow_run:" in workflow
    assert "workflow_dispatch:" in workflow

def test_failure_recovery_uses_workflow_scoped_queries_and_six_hour_window() -> None:
    workflow = _read(".github/workflows/action-failure-recovery.yml")
    assert 'gh run list --repo "${GH_REPO}" --workflow "${workflow}"' in workflow
    assert "max_age_seconds=21600" in workflow
    assert "run_attempt" in workflow
    assert "set -euo pipefail" in workflow
    assert "exit 1" in workflow

def test_failure_recovery_covers_all_world_model_lanes() -> None:
    workflow = _read(".github/workflows/action-failure-recovery.yml")
    for marker in (
        "Soccer Prospective In-Play PIT Capture",
        "Soccer Prospective In-Play Maturity",
        "Soccer World Model Supervisor",
        "Soccer Dynamic Simulator Research",
        "Soccer Match State Research",
        "Soccer Automation Integrity Audit",
    ):
        assert marker in workflow

def test_world_model_supervisor_fails_closed_and_parses_multi_document_evidence() -> None:
    workflow = _read(".github/workflows/soccer-world-model-supervisor.yml")
    assert "dispatch_failures=0" in workflow
    assert "dispatch_failures=$((dispatch_failures + 1))" in workflow
    assert 'if [ "$dispatch_failures" -gt 0 ]; then' in workflow
    assert "world_model_dispatch_failures" in workflow
    assert "JSONDecoder" in workflow
    assert "raw_decode" in workflow
    assert "exit 1" in workflow

def test_maturity_pr_validation_uses_exact_pr_head_and_never_persists_pr_changes() -> None:
    workflow = _read(".github/workflows/soccer-prospective-inplay-maturity.yml")
    assert "pull_request:" in workflow
    assert "github.event_name == 'pull_request' && github.event.pull_request.head.sha || 'main'" in workflow
    assert "github.event_name != 'pull_request'" in workflow
    assert "python -m pip install" in workflow
    assert "requirements.txt" in workflow

def test_ci_is_bounded_and_validates_workflow_yaml() -> None:
    workflow = _read(".github/workflows/ci.yml")
    assert "timeout --signal=TERM --kill-after=30s 300" in workflow
    assert "python -m pytest -q -vv --durations=20" in workflow
    assert "Validate workflow YAML syntax" in workflow
    assert 'Dir[".github/workflows/*.yml"].sort.each' in workflow
    for generated_path in (
        "data/research/prospective_inplay/raw/**",
        "data/research/match_state_snapshots/**",
        "data/research/match_state_research_status.json",
    ):
        assert generated_path in workflow

def test_controller_is_event_driven_and_same_repo_hardening_only() -> None:
    workflow = _read(".github/workflows/autonomous-github-controller.yml")
    assert "Soccer Automation Integrity Audit" in workflow
    assert "gh pr update-branch" in workflow
    assert "--rebase" in workflow
    assert '[[ "${head_ref}" == hardening/* ]]' in workflow
    assert '[ "${head_repo}" = "${GH_REPO}" ]' in workflow
    assert '[[ "${title,,}" == hardening:* ]]' in workflow
    assert "hardening-safe-v1" in workflow
    assert "gh pr merge" in workflow
    assert "--match-head-commit" in workflow
    for path in (
        ".github/workflows/action-failure-recovery.yml",
        ".github/workflows/ci.yml",
        ".github/workflows/soccer-automation-integrity-audit.yml",
        ".github/workflows/soccer-prospective-inplay-maturity.yml",
        ".github/workflows/soccer-world-model-supervisor.yml",
    ):
        assert path in workflow
