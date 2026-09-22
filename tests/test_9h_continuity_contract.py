from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTONOMOUS = ROOT / ".github" / "workflows" / "soccer-9h-autonomous.yml"
RECOVERY = ROOT / ".github" / "workflows" / "soccer-9h-recovery.yml"


def test_autonomous_workflow_has_continuous_9h_cycle_and_strict_concurrency():
    text = AUTONOMOUS.read_text(encoding="utf-8")
    assert "schedule:" in text
    assert 'cron: "15 0,8,16 * * *"' in text
    assert "workflow_dispatch:" in text
    assert "group: soccer-9h-autonomous-main" in text
    assert "cancel-in-progress: false" in text
    assert text.count("timeout-minutes: 180") >= 3
    for phase in ("phase1_gate:", "phase2_research:", "phase3_verification:"):
        assert phase in text
    assert "production_provenance.json" in text
    assert 'PIT_OPENFOOTBALL_COUNTRY_MAX_COMMIT_PAGES: "12"' in text


def test_recovery_workflow_has_watchdog_and_self_chaining_dispatch():
    text = RECOVERY.read_text(encoding="utf-8")
    assert 'cron: "2,7,12,17,22,27,32,37,42,47,52,57 * * * *"' in text
    assert "types: [completed]" in text
    assert "actions: write" in text
    assert "contents: read" in text
    assert "group: soccer-9h-recovery" in text
    assert "cancel-in-progress: true" in text
    assert 'if run.get("head_sha") == main_sha:' in text
    assert "rerun-failed-jobs" in text
    assert "/actions/workflows/soccer-9h-autonomous.yml/dispatches" in text
    assert "case" in text
    assert "DISPATCH)" in text
