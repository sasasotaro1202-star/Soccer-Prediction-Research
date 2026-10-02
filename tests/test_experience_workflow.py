from pathlib import Path


WORKFLOW = Path(".github/workflows/soccer-experience-ledger.yml")


def test_experience_workflow_fails_closed_on_schema_block_without_blocking_warmup():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'status") == "BLOCKED_LEDGER_SCHEMA"' in text
    assert 'raise SystemExit("experience learning is blocked by invalid ledger schema")' in text
    assert 'raise SystemExit("target-specific experience learning is blocked by invalid ledger schema")' in text
    assert 'status=\\{status.get(\'status\')\\}' not in text
