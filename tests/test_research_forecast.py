import pandas as pd

from src.prediction.research_forecast import verify


def test_research_forecast_verification_accepts_full_contract(tmp_path):
    path = tmp_path / "forecast.csv"
    pd.DataFrame([{
        "match_id": "test-1",
        "kickoff_utc": "2030-01-01T12:00:00+00:00",
        "competition": "EPL",
        "home_team": "Home FC",
        "away_team": "Away FC",
        "result_prediction": "Home FC",
        "result_prediction_probability": 0.50,
        "mom_method": "test",
        "home_win_probability": 0.50,
        "draw_probability": 0.20,
        "away_win_probability": 0.30,
        "score_1": "1-0",
        "score_1_probability": 0.20,
        "score_2": "1-1",
        "score_2_probability": 0.15,
        "score_3": "2-0",
        "score_3_probability": 0.10,
        "mom_status": "PREDICTED_HEURISTIC",
        "mom_1_player": "A",
        "mom_1_probability": 0.40,
        "mom_2_player": "B",
        "mom_2_probability": 0.30,
        "mom_3_player": "C",
        "mom_3_probability": 0.20,
        "mom_4_player": "D",
        "mom_4_probability": 0.10,
    }]).to_csv(path, index=False)

    assert verify(str(path))["status"] == "VERIFIED"


def test_research_forecast_verification_accepts_empty_target_set(tmp_path):
    from src.prediction.research_forecast import FORECAST_COLUMNS

    path = tmp_path / "forecast.csv"
    pd.DataFrame(columns=FORECAST_COLUMNS).to_csv(path, index=False)

    assert verify(str(path)) == {
        "status": "VERIFIED",
        "rows": 0,
        "empty_target_set": True,
    }


