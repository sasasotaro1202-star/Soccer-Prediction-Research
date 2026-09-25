from pathlib import Path
import re


def _has_immutable_action_pin(workflow: str, action: str) -> bool:
    pattern = rf"uses:\s+{re.escape(action)}@[0-9a-f]{{40}}(?:\s|$)"
    return re.search(pattern, workflow) is not None


def test_matchday_refresh_contract_is_read_only_and_quarter_hourly():
    content = Path(".github/workflows/soccer-matchday-intelligence.yml").read_text(encoding="utf-8")
    assert 'cron: "7,22,37,52 * * * *"' in content
    assert "push:" in content
    assert "workflow_run:" in content
    assert 'workflows: ["Soccer 9H Autonomous Research"]' in content
    assert "contents: read" in content
    assert "cancel-in-progress: false" in content
    assert _has_immutable_action_pin(content, "actions/upload-artifact")
    assert "path: |" in content
    assert "artifacts/future_matchday_fixtures.csv" in content
