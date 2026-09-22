import pandas as pd

from src.research.score_model_selection import select_score_model, verify_selected_score_model


def _frame(status="PASS"):
    base = pd.DataFrame({
        "n": [500, 500, 500],
        "score_logloss": [1.00, 1.01, 0.99],
        "over_2_5_logloss": [0.70, 0.71, 0.69],
        "over_2_5_brier": [0.20, 0.21, 0.19],
        "btts_logloss": [0.68, 0.69, 0.67],
        "btts_brier": [0.21, 0.22, 0.20],
        "negative_binomial_score_logloss": [0.94, 0.95, 0.93],
        "negative_binomial_over_2_5_logloss": [0.699, 0.709, 0.689],
        "negative_binomial_over_2_5_brier": [0.199, 0.209, 0.189],
        "negative_binomial_btts_logloss": [0.679, 0.689, 0.669],
        "negative_binomial_btts_brier": [0.209, 0.219, 0.199],
        "negative_binomial_status": [status] * 3,
    })
    return base


def test_negative_binomial_locked_status_is_fail_closed():
    development = _frame("PASS")
    selection = select_score_model(development)
    assert selection["selected_method"] == "negative_binomial"

    locked = development.copy()
    locked["negative_binomial_status"] = ["ERROR", "PASS", "PASS"]
    result = verify_selected_score_model(selection, locked)

    assert result["status"] == "REJECT"
    assert result["selected_method"] == "primary"
    assert result["reason"] == "negative_binomial is unavailable on locked OOS"


def test_negative_binomial_locked_status_passes_when_available():
    development = _frame("PASS")
    selection = select_score_model(development)
    assert selection["selected_method"] == "negative_binomial"

    locked = development.copy()
    result = verify_selected_score_model(selection, locked)

    assert result["status"] == "PASS"
    assert result["selected_method"] == "negative_binomial"
