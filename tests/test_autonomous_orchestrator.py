from pathlib import Path


WORKFLOW = Path(".github/workflows/soccer-autonomous-orchestrator.yml")


def test_orchestrator_is_fail_closed_and_main_pinned():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "permissions:" in text
    assert "actions: write" in text
    assert "contents: read" in text
    assert 'gh api "repos/${GH_REPO}/git/ref/heads/main"' in text
    assert 'gh workflow run "${workflow}" --repo "${GH_REPO}" --ref main' in text
    assert 'and r.get("headSha") == main_sha' in text
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



def test_ci_pr_verification_uses_head_sha_and_survives_merge_ref_churn():
    ci = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "github.event.pull_request.head.sha || github.sha" in ci
    assert "github.event_name == 'push'" in ci
