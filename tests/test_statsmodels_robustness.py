import pandas as pd

from src.research.external_oss.statsmodels_robustness import evaluate_statsmodels_robustness


def test_robustness_passes_when_fold_and_slice_are_non_regressed():
    result = evaluate_statsmodels_robustness(
        {
            "status": "RESEARCH_OOS_READY",
            "rows": [
                {
                    "pit_training_boundary_valid": True,
                    "pit_oos_case_valid": True,
                    "same_kickoff_split_avoided": True,
                    "calibration_precedes_oos": True,
                    "common_coverage": 0.95,
                    "unsupported_rows": 5,
                    "incumbent_calibrated_score_logloss": 1.0,
                    "statsmodels_calibrated_score_logloss": 0.99,
                    "competition_slices": [
                        {
                            "competition": "A",
                            "n": 200,
                            "incumbent_calibrated_score_logloss": 1.0,
                            "statsmodels_calibrated_score_logloss": 1.0,
                        },
                        {
                            "competition": "B",
                            "n": 150,
                            "incumbent_calibrated_score_logloss": 1.0,
                            "statsmodels_calibrated_score_logloss": 0.99,
                        }
                    ],
                },
                {
                    "pit_training_boundary_valid": True,
                    "pit_oos_case_valid": True,
                    "same_kickoff_split_avoided": True,
                    "calibration_precedes_oos": True,
                    "common_coverage": 0.90,
                    "unsupported_rows": 10,
                    "incumbent_calibrated_score_logloss": 1.0,
                    "statsmodels_calibrated_score_logloss": 1.0,
                    "competition_slices": [],
                },
                {
                    "pit_training_boundary_valid": True,
                    "pit_oos_case_valid": True,
                    "same_kickoff_split_avoided": True,
                    "calibration_precedes_oos": True,
                    "common_coverage": 0.92,
                    "unsupported_rows": 8,
                    "incumbent_calibrated_score_logloss": 1.0,
                    "statsmodels_calibrated_score_logloss": 1.0,
                    "competition_slices": [],
                },
            ],
        },
        min_folds=3,
        min_competition_rows=100,
    )
    assert result["status"] == "PASS"
    assert result["selection_performed"] is False
    assert result["frozen_holdout_used"] is False


def test_robustness_blocks_invalid_integrity():
    result = evaluate_statsmodels_robustness(
        {
            "status": "RESEARCH_OOS_READY",
            "rows": [
                {
                    "pit_training_boundary_valid": False,
                    "pit_oos_case_valid": True,
                    "same_kickoff_split_avoided": True,
                    "calibration_precedes_oos": True,
                    "incumbent_calibrated_score_logloss": 1.0,
                    "statsmodels_calibrated_score_logloss": 0.9,
                }
            ] * 3,
        }
    )
    assert result["status"] == "HOLD"
