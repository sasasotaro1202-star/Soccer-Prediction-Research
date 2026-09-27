import numpy as np
import pandas as pd

from src.prediction.secondary_outputs import fit_xg_score_rate_model, predict_xg_score_distribution
from src.research.score_model_selection import select_score_model


def test_xg_score_model_uses_only_pit_verified_xg_rows():
    history = pd.DataFrame([
        {"home_team":"A","away_team":"B","home_xg":1.8,"away_xg":0.7,"pit_verified":True,"competition":"EPL"},
        {"home_team":"B","away_team":"A","home_xg":0.9,"away_xg":1.1,"pit_verified":True,"competition":"EPL"},
        {"home_team":"A","away_team":"B","home_xg":99.0,"away_xg":0.01,"pit_verified":False,"competition":"EPL"},
    ])
    model = fit_xg_score_rate_model(history)
    assert model["training_rows"] == 2
    dist = predict_xg_score_distribution(model, "A", "B", "EPL")
    assert len(dist) == 169
    assert abs(sum(p for _, _, p in dist) - 1.0) < 1e-12


def test_xg_method_is_available_to_selection_when_oos_metrics_pass():
    rows=[]
    for i in range(3):
        rows.append({
            "n":600,
            "score_logloss":1.0,
            "over_2_5_logloss":0.70,
            "over_2_5_brier":0.20,
            "btts_logloss":0.70,
            "btts_brier":0.20,
            "xg_score_logloss":0.99,
            "xg_over_2_5_logloss":0.69,
            "xg_over_2_5_brier":0.195,
            "xg_btts_logloss":0.69,
            "xg_btts_brier":0.195,
            "xg_status":"PASS",
        })
    out=select_score_model(pd.DataFrame(rows),min_rows_per_block=500)
    assert out["selected_method"]=="xg"
