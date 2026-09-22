import pandas as pd
from src.research.score_model_selection import select_score_model

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
