from pathlib import Path


def test_experience_ledger_dependency_install_is_bounded_and_fail_closed():
    workflow = Path(".github/workflows/soccer-experience-ledger.yml").read_text(encoding="utf-8")
    assert "Install dependencies with bounded retry" in workflow
    assert "for attempt in 1 2 3; do" in workflow
    assert "sleep \$((attempt * 5))" in workflow
    assert 'if [ "\${attempt}" -eq 3 ]; then' in workflow
    assert "exit 1" in workflow
    assert "|| true" not in workflow


def test_experience_ledger_dependency_install_keeps_internal_network_bounds():
    workflow = Path(".github/workflows/soccer-experience-ledger.yml").read_text(encoding="utf-8")
    assert "--retries 8" in workflow
    assert "--timeout 300" in workflow
