from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTONOMOUS = ROOT / ".github" / "workflows" / "soccer-9h-autonomous.yml"
RECOVERY = ROOT / ".github" / "workflows" / "soccer-9h-recovery.yml"


def test_autonomous_workflow_has_continuous_9h_cycle_and_strict_concurrency():
    text = AUTONOMOUS.read_text(encoding="utf-8")
    assert "schedule:" in text
    assert "push:" in text
    assert '"src/**"' in text
    assert '"tests/**"' in text
    assert 'cron: "15 0,8,16 * * *"' in text
    assert "workflow_dispatch:" in text
    assert "\n  push:\n" in text
    assert "Recovery controller" in text or "recovery controller" in text.lower()
    assert "group: soccer-9h-autonomous-main" in text
    assert "cancel-in-progress: false" in text
    assert text.count("timeout-minutes: 180") >= 3
    for phase in ("phase1_gate:", "phase2_research:", "phase3_verification:"):
        assert phase in text
    assert "production_provenance.json" in text
    assert "pit_preflight_manifest.json" in text
    assert 'PIT_OPENFOOTBALL_COUNTRY_MAX_COMMIT_PAGES: "12"' in text
    assert "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803" in text
    assert "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1" in text
    assert "actions/cache@caa296126883cff596d87d8935842f9db880ef25" in text
    assert "actions/upload-artifact@b7c566a772e6b6bfb58ed0dc250532a479d7789f" in text
    assert text.count("retention-days: 3") == 3
    assert "if-no-files-found: error" in text


def test_recovery_workflow_has_watchdog_and_self_chaining_dispatch():
    text = RECOVERY.read_text(encoding="utf-8")
    assert 'cron: "13,28,43,58 * * * *"' in text
    assert "workflow_run:" in text
    assert 'workflows: ["Soccer 9H Autonomous Research"]' in text
    assert "types: [completed]" in text
    assert "actions: write" in text
    assert "contents: read" in text
    assert "group: soccer-9h-recovery-main" in text
    assert "cancel-in-progress: false" in text
    assert 'if run.get("head_sha") == main_sha:' in text
    assert "rerun-failed-jobs" in text
    assert "/actions/workflows/soccer-9h-autonomous.yml/dispatches" in text
    assert "case" in text
    assert "DISPATCH)" in text
    assert "cancel_duplicate_current_main_runs" in text
    assert "one canonical run retained" in text
    assert "?per_page=100" in text
    assert "max_active_age_hours = 10.5" in text
    assert "cancel_stale_current_main_runs" in text
    assert "before fresh dispatch" in text
    assert text.count("verify_current_main_active") >= 3
    assert "/tmp/verify-9h-runs.json" in text
    assert 'python - "${VERIFY_JSON}"' not in text
    assert '"requested"' in text
    assert "no active current-main run was observed" in text
    assert '{"pending", "queued", "waiting", "requested", "in_progress"}' in text


def test_phase3_runs_for_independent_verification_after_phase2_failure():
    text = AUTONOMOUS.read_text(encoding="utf-8")
    assert "if: always() && needs.phase2_research.result != 'cancelled'" in text
    assert "independent verification and robustness" in text


def test_recovery_preserves_active_obsolete_9h_run_on_main_advance():
    text = RECOVERY.read_text(encoding="utf-8")
    assert 'run.get("status") not in {"pending", "queued", "waiting", "requested"}' in text
    assert '"in_progress"' not in text.split("cancel_obsolete_runs()", 1)[1].split("cancel_stale_current_main_runs()", 1)[0]
    assert "Never interrupt an actively executing immutable research snapshot" in text
