from pathlib import Path


WORKFLOW = Path(".github/workflows/soccer-opta-like-24h.yml").read_text(encoding="utf-8")


def test_24h_has_success_only_phase_checkpoints():
    for phase in range(1, 5):
        assert f"phase{phase}_success.json" in WORKFLOW
    assert WORKFLOW.count("if: success()") >= 4


def test_24h_reconciler_requires_exact_success_handoffs():
    assert "Require success-only phase handoffs" in WORKFLOW
    assert "status_not_success" in WORKFLOW
    assert "research_only_contract" in WORKFLOW
    assert "production_changed_contract" in WORKFLOW
    assert "sha_mismatch" in WORKFLOW
    assert "run_id_mismatch" in WORKFLOW
