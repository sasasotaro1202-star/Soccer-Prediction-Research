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
        "over_2_5_probability": 0.55,
        "under_2_5_probability": 0.45,
        "btts_yes_probability": 0.48,
        "btts_no_probability": 0.52,
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




def test_daily_research_forecast_defers_without_adopted_model(tmp_path, monkeypatch):
    from src.prediction.research_forecast import FORECAST_COLUMNS, run

    fixtures = tmp_path / "fixtures.csv"
    output = tmp_path / "forecast.csv"
    status = tmp_path / "status.json"
    pd.DataFrame([{
        "match_id": "m1",
        "kickoff_utc": "2030-01-01T12:00:00Z",
        "competition": "EPL",
        "home_team": "Home FC",
        "away_team": "Away FC",
    }]).to_csv(fixtures, index=False)

    monkeypatch.chdir(tmp_path)
    result = run(
        str(fixtures),
        str(output),
        str(status),
        "2029-12-31T12:00:00Z",
    )

    assert result["status"] == "DEFERRED_NO_ADOPTED_MODEL"
    assert result["prediction_rows"] == 0
    frame = pd.read_csv(output)
    assert list(frame.columns) == list(FORECAST_COLUMNS)
    assert frame.empty


def test_verify_preserves_canonical_run_status_and_rows(tmp_path, monkeypatch):
    import json
    import sys
    import src.prediction.research_forecast as module

    status_path = tmp_path / "status.json"
    module.run = lambda *args: {
        "status": "PREDICTED_PRODUCTION_ADOPTED",
        "prediction_time_utc": "2030-01-01T00:00:00Z",
        "rows": 2,
        "prediction_rows": 2,
    }
    monkeypatch.setattr(module, "verify", lambda path: {
        "status": "VERIFIED",
        "rows": 2,
    })
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "research_forecast",
            "--fixtures", str(tmp_path / "fixtures.csv"),
            "--output", str(tmp_path / "forecast.csv"),
            "--status", str(status_path),
            "--verify",
        ],
    )

    module.main()

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "PREDICTED_PRODUCTION_ADOPTED"
    assert payload["rows"] == 2
    assert payload["prediction_rows"] == 2
    assert payload["output_contract_verified"] is True
    assert payload["output_contract_verification"]["status"] == "VERIFIED"


def test_daily_research_fallback_covers_non_active_competitions():
    from pathlib import Path
    import tempfile
    from src.prediction.research_forecast import run

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fixtures = root / "fixtures.csv"
        output = root / "forecast.csv"
        status = root / "status.json"
        pd.DataFrame([
            {
                "match_id": "active",
                "kickoff_utc": "2030-01-01T12:00:00Z",
                "competition": "EPL",
                "home_team": "Home FC",
                "away_team": "Away FC",
            },
            {
                "match_id": "research-only",
                "kickoff_utc": "2030-01-01T15:00:00Z",
                "competition": "BEL",
                "home_team": "Research Home",
                "away_team": "Research Away",
            },
        ]).to_csv(fixtures, index=False)

        result = run(
            str(fixtures),
            str(output),
            str(status),
            "2029-12-31T12:00:00Z",
        )

        frame = pd.read_csv(output)
        assert result["status"] == "PREDICTED_FALLBACK_BASELINE"
        assert set(frame["match_id"].astype(str)) == {"active", "research-only"}
