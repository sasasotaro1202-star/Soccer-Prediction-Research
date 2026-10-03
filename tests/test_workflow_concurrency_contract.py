from pathlib import Path


def test_verification_workflows_do_not_cancel_in_progress_runs():
    root = Path(__file__).resolve().parents[1]
    for relative in (
        ".github/workflows/ci.yml",
        ".github/workflows/soccer-rich-data-availability.yml",
    ):
        text = (root / relative).read_text(encoding="utf-8")
        assert "cancel-in-progress: false" in text, relative
        assert "cancel-in-progress: true" not in text, relative


def test_long_running_research_remains_non_canceling():
    root = Path(__file__).resolve().parents[1]
    text = (root / ".github/workflows/soccer-9h-autonomous.yml").read_text(encoding="utf-8")
    assert "cancel-in-progress: false" in text
