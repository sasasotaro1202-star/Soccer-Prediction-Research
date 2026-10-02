from pathlib import Path


def test_matchday_workflow_separates_scheduled_and_manual_modes():
    workflow = Path(".github/workflows/soccer-matchday-intelligence.yml").read_text(encoding="utf-8")
    assert 'prediction_mode:' in workflow
    assert 'default: "on_demand"' in workflow
    assert 'scheduled_1h)' in workflow
    assert 'on_demand)' in workflow
    assert "--prediction-window-minutes-before 60" in workflow
    assert "--prediction-window-tolerance-minutes 10" in workflow


def test_chat_gateway_routes_manual_matchday_requests_on_demand():
    workflow = Path(".github/workflows/soccer-chat-request-gateway.yml").read_text(encoding="utf-8")
    assert 'operation" = "matchday_intelligence"' in workflow
    assert '\"prediction_mode\":\"on_demand\"' in workflow
