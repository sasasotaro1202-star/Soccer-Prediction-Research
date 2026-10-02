from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "soccer-chat-request-gateway.yml"
DOC = ROOT / "docs" / "chat-execution-contract.md"

def test_chat_gateway_has_hard_short_runtime_bound():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "timeout-minutes: 3" in text
    assert "--connect-timeout 15 --max-time 45" in text

def test_chat_gateway_only_dispatches_allowlisted_workflows():
    text = WORKFLOW.read_text(encoding="utf-8")
    for operation in ("daily_forecast", "matchday_intelligence", "predictability_research", "research_9h"):
        assert operation in text
    assert "requests/inbox/*.json" in text
    assert "actions/workflows/" in text

def test_chat_gateway_is_idempotent_by_request_id():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "requests/state/${request_id}.json" in text
    assert "no duplicate dispatch" in text
    assert "\"status\":\"DISPATCHING\"" in text
    assert "\"status\":\"DISPATCHED\"" in text

def test_chat_gateway_does_not_hide_failures():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "|| true" not in text

def test_chat_execution_contract_forbids_sync_polling():
    text = DOC.read_text(encoding="utf-8")
    assert "without polling" in text
    assert "allow-listed" in text
    assert "PIT, OOS, holdout" in text

def test_chat_gateway_run_block_has_valid_yaml_indentation():
    lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
    in_run = False
    for line in lines:
        if line.startswith("        run: |"):
            in_run = True
            continue
        if in_run and line.startswith("      - name:"):
            in_run = False
            continue
        if in_run and line.strip():
            assert line.startswith("          "), line
