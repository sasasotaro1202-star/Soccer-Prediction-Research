import numpy as np
import pandas as pd

from src.prediction.live_research_forecast import lambdas, matrix, result_probs, verify


def test_lambdas_are_positive():
    home, away = lambdas(2070.0, 1947.0)
    assert home > away > 0
    assert np.isfinite([home, away]).all()


def test_result_probs_sum_to_one():
    probs = result_probs(matrix(1.6, 0.8))
    assert np.isfinite(probs).all()
    assert np.isclose(probs.sum(), 1.0)
    assert np.all((probs >= 0) & (probs <= 1))


def test_verify_rejects_production_status(tmp_path):
    path = tmp_path / "live.csv"
    pd.DataFrame([{
        "match_id":"x","kickoff_utc":"2030-01-01T00:00:00Z",
        "home_team":"France","away_team":"Belgium","result_prediction":"Home",
        "p_home":0.6,"p_draw":0.2,"p_away":0.2,
        "score_1":"1-0","score_1_probability":0.2,
        "score_2":"2-0","score_2_probability":0.15,
        "score_3":"1-1","score_3_probability":0.1,
        "prediction_state":"PREMATCH_RESEARCH_FORECAST",
        "pit_status":"CURRENT_OBSERVED_PRE_KICKOFF",
        "production_status":"PRODUCTION",
    }]).to_csv(path,index=False)
    try:
        verify(str(path))
    except RuntimeError as exc:
        assert "production flag" in str(exc)
    else:
        raise AssertionError("production status must fail closed")


def test_verify_accepts_research_contract(tmp_path):
    path = tmp_path / "live.csv"
    pd.DataFrame([{
        "match_id":"x","kickoff_utc":"2030-01-01T00:00:00Z",
        "home_team":"France","away_team":"Belgium","result_prediction":"Home",
        "p_home":0.6,"p_draw":0.2,"p_away":0.2,
        "score_1":"1-0","score_1_probability":0.2,
        "score_2":"2-0","score_2_probability":0.15,
        "score_3":"1-1","score_3_probability":0.1,
        "prediction_state":"PREMATCH_RESEARCH_FORECAST",
        "pit_status":"CURRENT_OBSERVED_PRE_KICKOFF",
        "production_status":"NOT_PRODUCTION",
    }]).to_csv(path,index=False)
    assert verify(str(path))["status"] == "VERIFIED"
