import pandas as pd

from src.research.adoption import adoption_decision


def _frames(rows, *, logloss=0.95, brier=0.24, rps=0.19):
    base = pd.DataFrame({
        "n": rows,
        "logloss": [1.0] * len(rows),
        "accuracy": [0.50] * len(rows),
        "brier": [0.25] * len(rows),
        "rps": [0.20] * len(rows),
        "ece": [0.10] * len(rows),
    })
    cand = base.copy()
    cand["logloss"] = logloss
    cand["accuracy"] = 0.52
    cand["brier"] = brier
    cand["rps"] = rps
    cand["ece"] = 0.09
    return base, cand


def _development(blocks=10, good_blocks=10):
    base = pd.DataFrame({
        "logloss": [1.0] * blocks,
        "accuracy": [0.50] * blocks,
        "brier": [0.25] * blocks,
        "rps": [0.20] * blocks,
        "ece": [0.10] * blocks,
    })
    cand = base.copy()
    for i in range(blocks):
        if i < good_blocks:
            cand.loc[i, ["logloss", "accuracy", "brier", "rps", "ece"]] = [0.95, 0.52, 0.24, 0.19, 0.09]
        else:
            cand.loc[i, ["logloss", "accuracy", "brier", "rps", "ece"]] = [1.01, 0.49, 0.251, 0.201, 0.11]
    cand["baseline_logistic_logloss"] = base["logloss"]
    cand["baseline_logistic_accuracy"] = base["accuracy"]
    cand["baseline_logistic_brier"] = base["brier"]
    cand["baseline_logistic_rps"] = base["rps"]
    cand["baseline_logistic_ece"] = base["ece"]
    return cand


def test_adoption_holds_when_locked_block_is_too_small():
    base, cand = _frames([500, 499])
    result = adoption_decision(base, cand, development_oos=_development())
    assert result["status"] == "HOLD"
    assert result["oos_claimed"] is False
    assert result["minimum_locked_rows_per_block"] == 500


def test_adoption_requires_three_percent_relative_logloss():
    base, cand = _frames([500, 500], logloss=0.971, brier=0.247)
    result = adoption_decision(base, cand, development_oos=_development())
    assert result["status"] == "REJECT"
    assert result["checks"]["primary_logloss_threshold_met"] is False


def test_adoption_requires_one_percent_auxiliary_improvement():
    base, cand = _frames([500, 500], logloss=0.96, brier=0.248, rps=0.199)
    result = adoption_decision(base, cand, development_oos=_development())
    assert result["status"] == "REJECT"
    assert result["checks"]["auxiliary_threshold_met"] is False


def test_adoption_allows_seventy_percent_development_non_regression():
    base, cand = _frames([500, 500], logloss=0.969, brier=0.247, rps=0.19)
    result = adoption_decision(base, cand, development_oos=_development(blocks=10, good_blocks=7))
    assert result["status"] == "ADOPT"
    assert result["development_stability"]["non_regressed_blocks"] == 7
    assert result["development_stability"]["non_regression_fraction"] == 0.7
    assert result["checks"]["primary_logloss_threshold_met"] is True
    assert result["checks"]["auxiliary_threshold_met"] is True
