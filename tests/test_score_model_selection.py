import pandas as pd
from src.research.score_model_selection import select_score_model, verify_selected_score_model

def _frame(better=True):
    return pd.DataFrame({
        "n":[100,100,100],
        "score_logloss":[1.00,1.02,0.98],
        "over_2_5_logloss":[0.70,0.72,0.71],
        "over_2_5_brier":[0.20,0.21,0.19],
        "btts_logloss":[0.68,0.69,0.67],
        "btts_brier":[0.21,0.22,0.20],
        "recency_score_logloss":[0.98,1.00,0.97] if better else [1.03,1.04,1.00],
        "recency_over_2_5_logloss":[0.699,0.719,0.709],
        "recency_over_2_5_brier":[0.199,0.209,0.189],
        "recency_btts_logloss":[0.679,0.689,0.669],
        "recency_btts_brier":[0.209,0.219,0.199],
        "dc_score_logloss":[1.01,1.01,0.99],
        "dc_over_2_5_logloss":[0.701,0.721,0.711],
        "dc_over_2_5_brier":[0.201,0.211,0.191],
        "dc_btts_logloss":[0.681,0.691,0.671],
        "dc_btts_brier":[0.211,0.221,0.201],
        "recency_status":["PASS"]*3,
        "dc_status":["PASS"]*3,
    })

def test_selector_adopts_recency_from_development_oos():
    result=select_score_model(_frame(True))
    assert result["selected_method"]=="recency"
    assert result["selection_rule"]["locked_oos_inspected"] is False

def test_selector_can_choose_time_decay_challenger():
    frame = _frame(False).copy()
    frame["time_decay_score_logloss"] = [0.96, 0.97, 0.95]
    frame["time_decay_over_2_5_logloss"] = [0.699, 0.719, 0.709]
    frame["time_decay_over_2_5_brier"] = [0.199, 0.209, 0.189]
    frame["time_decay_btts_logloss"] = [0.679, 0.689, 0.669]
    frame["time_decay_btts_brier"] = [0.209, 0.219, 0.199]
    frame["time_decay_status"] = ["PASS"] * 3
    result = select_score_model(frame)
    assert result["selected_method"] == "time_decay"
    assert result["candidate_records"]["time_decay"]["status"] == "ACCEPT"


def test_selector_keeps_primary_when_challenger_is_worse():
    result=select_score_model(_frame(False))
    assert result["selected_method"]=="primary"
    assert result["status"]=="KEEP_PRIMARY"

def test_unavailable_challenger_does_not_block_primary():
    frame=_frame(True).drop(columns=["dc_score_logloss","dc_over_2_5_logloss","dc_over_2_5_brier","dc_btts_logloss","dc_btts_brier"])
    frame["dc_status"]=["ERROR"]*3
    result=select_score_model(frame)
    assert result["selected_method"]=="recency"
    assert result["candidate_records"]["dixon_coles"]["status"]=="UNAVAILABLE"


def test_locked_oos_can_confirm_development_selected_recency():
    development = _frame(True)
    selection = select_score_model(development)
    locked = development.copy()
    result = verify_selected_score_model(selection, locked)
    assert result["status"] == "PASS"
    assert result["selected_method"] == "recency"
    assert result["locked_oos_inspected"] is True


def test_locked_oos_rejects_challenger_that_regresses():
    development = _frame(True)
    selection = select_score_model(development)
    locked = development.copy()
    locked["recency_score_logloss"] = [1.20, 1.20, 1.20]
    result = verify_selected_score_model(selection, locked)
    assert result["status"] == "REJECT"
    assert result["selected_method"] == "primary"


def test_primary_locked_oos_requires_all_market_metrics():
    development = _frame(True)
    selection = select_score_model(development)
    locked = development.drop(columns=["btts_brier"])
    result = verify_selected_score_model(selection, locked)
    assert result["status"] == "REJECT"
    assert "btts_brier" in ",".join(result["missing_columns"])


def test_primary_locked_oos_returns_explicit_market_evidence():
    development = _frame(True)
    selection = select_score_model(development)
    locked = development.copy()
    result = verify_selected_score_model(selection, locked)
    assert result["status"] == "PASS"
    assert result["selected_method"] == "primary"
    assert result["locked_oos_inspected"] is True
    assert result["checks"]["all_required_locked_metrics_finite"] is True


def test_selector_can_choose_negative_binomial_challenger():
    frame = _frame(False).copy()
    frame["negative_binomial_score_logloss"] = [0.94, 0.95, 0.93]
    frame["negative_binomial_over_2_5_logloss"] = [0.699, 0.719, 0.709]
    frame["negative_binomial_over_2_5_brier"] = [0.199, 0.209, 0.189]
    frame["negative_binomial_btts_logloss"] = [0.679, 0.689, 0.669]
    frame["negative_binomial_btts_brier"] = [0.209, 0.219, 0.199]
    frame["negative_binomial_status"] = ["PASS"] * 3
    result = select_score_model(frame)
    assert result["selected_method"] == "negative_binomial"
    assert result["candidate_records"]["negative_binomial"]["status"] == "ACCEPT"
