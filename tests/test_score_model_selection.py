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
        "recency_d365_score_logloss":[0.98,1.00,0.97] if better else [1.03,1.04,1.00],
        "recency_d365_over_2_5_logloss":[0.699,0.719,0.709],
        "recency_d365_over_2_5_brier":[0.199,0.209,0.189],
        "recency_d365_btts_logloss":[0.679,0.689,0.669],
        "recency_d365_btts_brier":[0.209,0.219,0.199],
        "recency_d365_status":["PASS"]*3,
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
    assert result["selection_rule"]["min_development_blocks"] == 3

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


def test_primary_locked_oos_emits_explicit_market_evidence():
    selection = select_score_model(_frame(True))
    locked = _frame(True)
    result = verify_selected_score_model(selection={"selected_method": "primary"}, locked_oos=locked)
    assert result["status"] == "PASS"
    assert result["market_metrics_finite"] is True
    assert result["locked_oos_blocks"] == 3
    assert result["locked_oos_rows"] == 300


def test_primary_locked_oos_rejects_non_finite_btts_evidence():
    locked = _frame(True)
    locked.loc[0, "btts_brier"] = float("nan")
    result = verify_selected_score_model(selection={"selected_method": "primary"}, locked_oos=locked)
    assert result["status"] == "REJECT"
    assert result["market_metrics_finite"] is False


def test_recency_half_life_is_selected_from_development_oos():
    frame = _frame(True).copy()
    for half in (180, 365, 730, 1095):
        for metric in ("score_logloss", "over_2_5_logloss", "over_2_5_brier", "btts_logloss", "btts_brier"):
            frame[f"recency_d{half}_{metric}"] = frame[f"recency_d365_{metric}"]
        frame[f"recency_d{half}_status"] = ["PASS"] * len(frame)
    frame["recency_d180_score_logloss"] = [0.90, 0.91, 0.89]
    result = select_score_model(frame)
    assert result["selected_method"] == "recency"
    assert result["selected_parameters"]["half_life_days"] == 180.0


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
    locked["recency_d365_score_logloss"] = [1.20, 1.20, 1.20]
    result = verify_selected_score_model(selection, locked)
    assert result["status"] == "REJECT"
    assert result["selected_method"] == "primary"
