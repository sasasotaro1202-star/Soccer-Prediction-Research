from pathlib import Path


WORKFLOW = Path(".github/workflows/pit-replay-audit.yml")


def test_pit_replay_workflow_has_bounded_dependency_retry():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "for attempt in 1 2 3; do" in text
    assert "--retries 8 --timeout 300" in text
    assert 'if [ "${attempt}" -eq 3 ]; then' in text


def test_pit_replay_workflow_does_not_hide_unexpected_failures():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "continue-on-error: true" not in text
    assert 'if [ "$code" -eq 2 ]; then' in text
    assert 'exit "$code"' in text


def test_pit_replay_workflow_treats_exit_2_as_blocked_diagnostic_only():
    text = WORKFLOW.read_text(encoding="utf-8")
    block = text.split("Run small PIT replay audit", 1)[1]
    block = block.split("Upload PIT audit artifacts", 1)[0]
    assert "PIT replay audit is BLOCKED" in block
    assert 'exit 0' in block
    assert 'exit "$code"' in block