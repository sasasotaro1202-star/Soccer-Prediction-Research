from pathlib import Path

WORKFLOW = Path(".github/workflows/soccer-historical-learning.yml")

def test_historical_learning_workflow_is_parallel_and_research_only():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'cron: "3 */6 * * *"' in text
    assert "group: soccer-historical-learning" in text
    assert "python -m src.data.completion_gate" in text
    assert "python -m src.research.engine" in text
    assert "production_changed" in text
    assert "production_model_path_touched" in text
    assert "permissions:\n  contents: read" in text
    assert "git push" not in text
    assert "models/current" not in text