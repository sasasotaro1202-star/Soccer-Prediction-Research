from pathlib import Path
import re


def _has_immutable_action_pin(workflow: str, action: str) -> bool:
    pattern = rf"uses:\s+{re.escape(action)}@[0-9a-f]{{40}}(?:\s|$)"
    return re.search(pattern, workflow) is not None


def test_matchday_refresh_contract_is_read_only_and_quarter_hourly():
    content = Path(".github/workflows/soccer-matchday-intelligence.yml").read_text(encoding="utf-8")
    assert 'cron: "7,22,37,52 * * * *"' in content
    assert "push:" not in content
    assert "workflow_run:" in content
    assert 'workflows: ["Soccer 9H Autonomous Research"]' in content
    assert "contents: read" in content
    assert "cancel-in-progress: false" in content
    assert _has_immutable_action_pin(content, "actions/upload-artifact")
    assert "path: |" in content
    assert "artifacts/future_matchday_fixtures.csv" in content


def test_matchday_prediction_execution_requires_adopted_bundle_or_total_fallback():
    content = Path(".github/workflows/soccer-matchday-intelligence.yml").read_text(encoding="utf-8")
    assert 'Run current predictions when an adopted production model exists' in content
    assert '[ -s models/current/production_model.pkl ] && [ -s models/current/production_model.json ]' in content
    assert '&& [ -s models/current/model_registry.json ] && [ -s models/current/production_provenance.json ]' in content
    assert "--model-policy production" in content
    assert 'No adopted production model present; routing to research-only total fallback.' in content
    assert "python -m src.prediction.research_forecast" in content
    assert '--output artifacts/predictions.csv' in content
    assert '--status artifacts/prediction_status.json' in content


def test_matchday_fallback_does_not_depend_on_model_only_feature_preparation():
    content = Path(".github/workflows/soccer-matchday-intelligence.yml").read_text(encoding="utf-8")
    assert "Prepare PIT-safe future fixture features when adopted production model exists" in content
    assert "PREPARE_SKIPPED_FOR_FALLBACK" in content
    assert "research-only total fallback consumes the validated fixture snapshot directly." in content
    assert '--fixtures artifacts/future_matchday_fixtures.csv' in content
