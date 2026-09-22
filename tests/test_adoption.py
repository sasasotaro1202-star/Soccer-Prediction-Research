import pandas as pd

from src.research.adoption import adoption_decision


def _frames(rows):
    base = pd.DataFrame({
        "n": rows,
        "logloss": [1.0] * len(rows),
        "accuracy": [0.50] * len(rows),
        "brier": [0.25] * len(rows),
        "rps": [0.20] * len(rows),
        "ece": [0.10] * len(rows),
    })
    cand = base.copy()
    cand["logloss"] = 0.95
    cand["accuracy"] = 0.52
    cand["brier"] = 0.24
    cand["rps"] = 0.19
    cand["ece"] = 0.09
    return base, cand


def test_adoption_holds_when_locked_block_is_too_small():
    base, cand = _frames([500, 499])
    result = adoption_decision(base, cand)
    assert result["status"] == "HOLD"
    assert result["oos_claimed"] is False
    assert result["minimum_locked_rows_per_block"] == 500
