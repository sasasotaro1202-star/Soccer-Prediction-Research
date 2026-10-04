from pathlib import Path


WORKFLOW = Path(".github/workflows/autonomous-github-controller.yml")


def test_autonomous_orchestrator_is_explicitly_allowlisted():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert ".github/workflows/soccer-autonomous-orchestrator.yml" in text
    assert "scripts/experience_commit_guard.py" in text
    assert "automation_policy: hardening-safe-v1" in text
    assert "gh pr merge" in text


def test_production_affecting_paths_remain_outside_allowlist():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "models/current" not in text
    assert "src/features/" not in text
    assert "src/models/" not in text
    assert "frozen_holdout" in text


def test_safe_hardening_prs_may_be_retargeted_but_research_prs_are_not():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'repos/${GH_REPO}/pulls/${number}' in text
    assert 'automation_policy: hardening-safe-v1' in text
    assert '-f base=main' in text
