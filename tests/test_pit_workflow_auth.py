from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pit_workflows_export_github_token():
    workflow_paths = (
        REPO_ROOT / ".github/workflows/soccer-coverage-pit-audit.yml",
        REPO_ROOT / ".github/workflows/soccer-research-robust.yml",
    )
    for path in workflow_paths:
        text = path.read_text(encoding="utf-8")
        assert "GITHUB_TOKEN: ${{ github.token }}" in text


def test_coverage_pit_workflow_exports_token_at_job_scope():
    path = REPO_ROOT / ".github/workflows/soccer-coverage-pit-audit.yml"
    text = path.read_text(encoding="utf-8")
    assert "permissions:\n  contents: read\n\nenv:\n" in text