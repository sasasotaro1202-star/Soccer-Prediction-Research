def test_mom_research_runner_exposes_calibrated_evaluator():
    import scripts.run_mom_research as runner

    assert callable(runner.run_mom_walk_forward_calibrated_soft_ensemble)
