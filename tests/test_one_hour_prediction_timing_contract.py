from pathlib import Path


def test_matchday_workflow_targets_one_hour_before_kickoff():
    workflow = Path(".github/workflows/soccer-matchday-intelligence.yml").read_text(encoding="utf-8")
    assert "--prediction-window-minutes-before 60" in workflow
    assert "--prediction-window-tolerance-minutes 10" in workflow


def test_runner_soft_window_recovery_contract_is_explicit():
    source = Path("src/prediction/runner.py").read_text(encoding="utf-8")
    assert 'mode": "SOFT_TARGET"' in source
    assert 'deadline_minutes_before": upper' in source
    assert 'prediction_timing_class' in source
