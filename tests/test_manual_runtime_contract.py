from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

def test_manual_user_facing_workflows_use_full_job_budget():
    expected = {
        "soccer-daily-research-forecast.yml": "timeout-minutes: ${{ github.event_name == 'workflow_dispatch' && 360 || 60 }}",
        "soccer-matchday-intelligence.yml": "timeout-minutes: ${{ github.event_name == 'workflow_dispatch' && 360 || 20 }}",
        "soccer-predictability-research.yml": "timeout-minutes: ${{ github.event_name == 'workflow_dispatch' && 360 || 45 }}",
        "soccer-versioned-pit-research.yml": "timeout-minutes: ${{ github.event_name == 'workflow_dispatch' && 360 || 45 }}",
        "pit-replay-audit.yml": "timeout-minutes: ${{ github.event_name == 'workflow_dispatch' && 360 || 120 }}",
        "soccer-coverage-pit-audit.yml": "timeout-minutes: ${{ github.event_name == 'workflow_dispatch' && 360 || 180 }}",
        "soccer-mom-research.yml": "timeout-minutes: ${{ github.event_name == 'workflow_dispatch' && 360 || 240 }}",
        "soccer-research-robust.yml": "timeout-minutes: ${{ github.event_name == 'workflow_dispatch' && 360 || 240 }}",
        "innovative-control-v13.yml": "timeout-minutes: ${{ github.event_name == 'workflow_dispatch' && 360 || 300 }}",
        "innovative-control-v2.yml": "timeout-minutes: ${{ github.event_name == 'workflow_dispatch' && 360 || 180 }}",
        "soccer-global-datalake.yml": "timeout-minutes: ${{ github.event_name == 'workflow_dispatch' && 360 || 30 }}",
        "soccer-research-cycle.yml": "timeout-minutes: ${{ github.event_name == 'workflow_dispatch' && 360 || 120 }}",
        "soccer-experience-ledger.yml": "timeout-minutes: ${{ github.event_name == 'workflow_dispatch' && 360 || 45 }}",
    }
    for name, line in expected.items():
        text = (WORKFLOWS / name).read_text(encoding="utf-8")
        assert line in text


def test_core_research_workflows_do_not_cancel_inflight_runs():
    for name in (
        "innovative-control-v13.yml",
        "innovative-control-v2.yml",
        "soccer-global-datalake.yml",
        "soccer-research-cycle.yml",
        "soccer-experience-ledger.yml",
    ):
        text = (WORKFLOWS / name).read_text(encoding="utf-8")
        assert "cancel-in-progress: false" in text

def test_9h_manual_workflow_already_uses_hosted_job_limit_per_phase():
    text = (WORKFLOWS / "soccer-9h-autonomous.yml").read_text(encoding="utf-8")
    assert "timeout-minutes: 180" in text
    assert "timeout-minutes: 360" in text
    assert "phase2_research" in text


def test_long_running_research_workflows_do_not_cancel_inflight_runs():
    for name in (
        "soccer-versioned-pit-research.yml",
        "pit-replay-audit.yml",
        "soccer-coverage-pit-audit.yml",
        "soccer-mom-research.yml",
        "soccer-research-robust.yml",
    ):
        text = (WORKFLOWS / name).read_text(encoding="utf-8")
        assert "cancel-in-progress: false" in text

def test_predictability_never_cancels_inflight_manual_work():
    text = (WORKFLOWS / "soccer-predictability-research.yml").read_text(encoding="utf-8")
    assert "cancel-in-progress: false" in text

def test_chat_gateway_stays_short_only_for_acknowledgement():
    text = (WORKFLOWS / "soccer-chat-request-gateway.yml").read_text(encoding="utf-8")
    assert "timeout-minutes: 3" in text
    assert "--connect-timeout 15 --max-time 45" in text
    assert "without polling for completion" in text
