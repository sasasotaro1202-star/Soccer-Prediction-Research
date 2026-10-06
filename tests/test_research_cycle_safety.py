from pathlib import Path

WORKFLOW = Path(".github/workflows/soccer-research-cycle.yml")


def test_legacy_research_cycle_is_manual_only_and_safe_runner_backed():
    text = WORKFLOW.read_text(encoding="utf-8")
    trigger = text.split("permissions:", 1)[0]
    assert "workflow_dispatch:" in trigger
    assert "schedule:" not in trigger
    assert "run: python -m src.research.safe_runner" in text
    assert "run: python -m src.research.engine" not in text
    assert "permissions:\n  contents: read" in text
    assert "git push" not in text
    assert "git commit" not in text


def test_legacy_research_cycle_keeps_strict_preflight_before_engine():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "python -m pytest -q" in text
    assert "python -m src.data.fixture_field_audit" in text
    assert "python -m src.data.completion_gate" in text
    assert 'TESTS_PASSED: "true"' in text
    assert "AUDIT_PASSED: ${{ steps.handoff.outcome == 'success' }}" in text
