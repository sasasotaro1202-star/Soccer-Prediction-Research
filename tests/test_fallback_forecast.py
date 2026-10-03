from __future__ import annotations

import pandas as pd

from src.prediction.fallback_forecast import build_fallback_forecast, predict_fallback_fixture
from src.research.forecast_health import audit


def test_fallback_predicts_unseen_teams_with_global_prior():
    result = predict_fallback_fixture(
        home_team="Unknown Home U23",
        away_team="Unknown Away U23",
        competition="AG_M",
        history=pd.DataFrame(),
        prediction_time=pd.Timestamp("2026-10-03T10:00:00Z"),
        neutral_venue=True,
    )
    probs = [
        result["p_home"],
        result["p_draw"],
        result["p_away"],
    ]
    assert abs(sum(probs) - 1.0) < 1e-12
    assert 0.0 <= min(probs) <= max(probs) <= 1.0
    assert len(result["scores"]) == 3
    assert abs(result["over_2_5_probability"] + result["under_2_5_probability"] - 1.0) < 1e-12
    assert abs(result["btts_yes_probability"] + result["btts_no_probability"] - 1.0) < 1e-12


def test_fallback_history_is_strictly_cutoff_and_pit_filtered(tmp_path):
    history_path = tmp_path / "history.csv"
    pd.DataFrame([
        {
            "match_id": "before",
            "kickoff_utc": "2026-09-01T12:00:00Z",
            "source_available_at_utc": "2026-09-01T09:00:00Z",
            "home_team": "Alpha",
            "away_team": "Beta",
            "competition": "AG_M",
            "home_goals": 2,
            "away_goals": 0,
            "pit_verified": True,
        },
        {
            "match_id": "after",
            "kickoff_utc": "2026-10-03T11:00:00Z",
            "source_available_at_utc": "2026-10-03T09:00:00Z",
            "home_team": "Alpha",
            "away_team": "Beta",
            "competition": "AG_M",
            "home_goals": 0,
            "away_goals": 4,
            "pit_verified": True,
        },
        {
            "match_id": "late-source",
            "kickoff_utc": "2026-09-02T12:00:00Z",
            "source_available_at_utc": "2026-10-03T12:00:00Z",
            "home_team": "Gamma",
            "away_team": "Beta",
            "competition": "AG_M",
            "home_goals": 4,
            "away_goals": 0,
            "pit_verified": True,
        },
        {
            "match_id": "unverified",
            "kickoff_utc": "2026-09-03T12:00:00Z",
            "source_available_at_utc": "2026-09-03T09:00:00Z",
            "home_team": "Delta",
            "away_team": "Beta",
            "competition": "AG_M",
            "home_goals": 4,
            "away_goals": 0,
            "pit_verified": False,
        },
    ]).to_csv(history_path, index=False)

    fixtures = pd.DataFrame([{
        "match_id": "future",
        "kickoff_utc": "2026-10-03T12:00:00Z",
        "competition": "AG_M",
        "home_team": "Alpha",
        "away_team": "Beta",
    }])
    output, meta = build_fallback_forecast(
        fixtures,
        pd.Timestamp("2026-10-03T10:00:00Z"),
        history_path=str(history_path),
    )

    assert len(output) == 1
    assert meta["rows_pit_eligible"] == 1
    assert output.loc[0, "fallback_history_rows"] == 1
    assert output.loc[0, "fallback_competition_history_rows"] == 1
    assert output.loc[0, "result_prediction"] == "Alpha"
    assert abs(
        output.loc[0, "home_win_probability"]
        + output.loc[0, "draw_probability"]
        + output.loc[0, "away_win_probability"]
        - 1.0
    ) < 1e-12


def test_forecast_health_accepts_total_fallback_status(tmp_path):
    output_path = tmp_path / "forecast.csv"
    status_path = tmp_path / "status.json"
    from src.prediction.research_forecast import FORECAST_COLUMNS
    row = {c: "" for c in FORECAST_COLUMNS}
    row.update({
        "match_id": "m1",
        "kickoff_utc": "2030-01-01T12:00:00Z",
        "competition": "U23_M",
        "home_team": "Home U23",
        "away_team": "Away U23",
        "home_win_probability": 0.40,
        "draw_probability": 0.30,
        "away_win_probability": 0.30,
        "result_prediction": "Home U23",
        "result_prediction_probability": 0.40,
        "score_1": "1-0",
        "score_1_probability": 0.20,
        "score_2": "1-1",
        "score_2_probability": 0.15,
        "score_3": "0-0",
        "score_3_probability": 0.10,
        "over_2_5_probability": 0.55,
        "under_2_5_probability": 0.45,
        "btts_yes_probability": 0.48,
        "btts_no_probability": 0.52,
        "mom_status": "BLOCKED_UPSTREAM_PLAYER_MODEL",
        "mom_method": "fallback_total_coverage_no_player_probability_source",
    })
    pd.DataFrame([row]).to_csv(output_path, index=False)
    status_path.write_text(
        __import__("json").dumps({
            "status": "PREDICTED_FALLBACK_BASELINE",
            "prediction_time_utc": "2029-12-31T12:00:00Z",
            "rows": 1,
            "prediction_rows": 1,
            "production_model_used": False,
            "fallback_used": True,
            "research_only": True,
        }),
        encoding="utf-8",
    )

    report = audit(str(output_path), str(status_path))
    assert report["ok"] is True
    assert report["status"] == "HEALTHY"


def test_fallback_uses_only_explicitly_pit_verified_market_prior():
    from src.prediction.fallback_forecast import predict_fallback_fixture

    prediction_time = pd.Timestamp("2030-01-01T12:00:00Z")
    base = predict_fallback_fixture(
        home_team="Home",
        away_team="Away",
        competition="AG_M",
        history=pd.DataFrame(),
        prediction_time=prediction_time,
        neutral_venue=True,
    )
    market = predict_fallback_fixture(
        home_team="Home",
        away_team="Away",
        competition="AG_M",
        history=pd.DataFrame(),
        prediction_time=prediction_time,
        neutral_venue=True,
        market_probabilities=(0.80, 0.10, 0.10),
    )

    assert market["evidence"]["market_prior_used"] is True
    assert market["evidence"]["market_weight"] == 0.35
    assert market["p_home"] > base["p_home"]
    assert abs(
        market["p_home"] + market["p_draw"] + market["p_away"] - 1.0
    ) < 1e-12
