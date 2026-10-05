from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github" / "workflows"


def test_orchestrator_budget_and_active_cap_are_bounded():
    text = (WF / "soccer-autonomous-orchestrator.yml").read_text(encoding="utf-8")
    assert "dispatch_budget=4" in text
    assert "max_active_runs=4" in text
    assert 'print("- dispatch_budget: 4")' in text
    assert 'print("- max_active_runs: 4")' in text


def test_control_plane_watchdog_has_queue_grace():
    text = (WF / "soccer-control-plane-watchdog.yml").read_text(encoding="utf-8")
    assert "queued_grace_min = 180" in text
    assert 'print("QUEUED_WAIT")' in text
    assert 'print("QUEUED_STALE")' in text


def test_9h_queue_watchdog_uses_long_prestart_grace():
    text = (WF / "soccer-9h-queue-watchdog.yml").read_text(encoding="utf-8")
    assert 'age_min >= 120.0' in text
    assert 'age_min >= 20.0' not in text


def test_9h_recovery_uses_long_prestart_grace():
    text = (WF / "soccer-9h-recovery.yml").read_text(encoding="utf-8")
    assert "120+ minutes" in text
    assert "stuck_minutes = 120" in text
    assert "(120.0 / 60.0)" in text
    assert "(20.0 / 60.0)" not in text
