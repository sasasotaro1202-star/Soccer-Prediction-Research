import json

import numpy as np
import pandas as pd

from src.prediction import runner


def test_missing_mom_input_does_not_block_primary_outputs(tmp_path, monkeypatch):
    now = pd.Timestamp("2030-01-01T00:00:00Z")
    fixtures = pd.DataFrame(
        [
            {
                "match_id": "fx-1",
                "kickoff_utc": "2030-01-01T03:00:00Z",
                "home_team": "A",
                "away_team": "B",
                "competition": "EPL",
                "source_available_at_utc": "2029-12-31T20:00:00Z",
                "pit_verified": True,
                "starter_status": "CONFIRMED",
            }
        ]
    )
    fixtures_path = tmp_path / "future_fixtures.csv"
    fixtures.to_csv(fixtures_path, index=False)

    monkeypatch.setattr(
        runner,
        "load_adopted_model",
        lambda *_args, **_kwargs: {
            "adoption_status": "ADOPT",
            "model_version": "v1",
            "oos_verified": True,
        },
    )
    monkeypatch.setattr(
        runner,
        "load_bundle",
        lambda *_args, **_kwargs: {
            "schema_version": 2,
            "model_version": "v1",
            "temperature": 1.0,
            "models": {},
            "weights": {},
            "score_model": {"method": "pit_smoothed_venue_split_team_goal_rates"},
        },
    )
    monkeypatch.setattr(
        runner,
        "predict_bundle",
        lambda *_args, **_kwargs: np.asarray([[0.50, 0.30, 0.20]], dtype=float),
    )
    monkeypatch.setattr(
        runner,
        "predict_score_candidates",
        lambda *_args, **_kwargs: [
            {"home_goals": 1, "away_goals": 0, "probability": 0.20, "rank": 1},
            {"home_goals": 1, "away_goals": 1, "probability": 0.15, "rank": 2},
            {"home_goals": 2, "away_goals": 1, "probability": 0.10, "rank": 3},
        ],
    )
    monkeypatch.setattr(
        runner,
        "predict_score_markets",
        lambda *_args, **_kwargs: {
            "over_0_5": 0.80, "under_0_5": 0.20,
            "over_1_5": 0.55, "under_1_5": 0.45,
            "over_2_5": 0.35, "under_2_5": 0.65,
            "over_3_5": 0.18, "under_3_5": 0.82,
            "over_4_5": 0.08, "under_4_5": 0.92,
            "btts_yes": 0.40, "btts_no": 0.60,
        },
    )

    status = runner.run(
        fixtures_path=str(fixtures_path),
        bundle_path=str(tmp_path / "bundle.pkl"),
        output_path=str(tmp_path / "predictions.csv"),
        status_path=str(tmp_path / "status.json"),
        prediction_time=now.isoformat(),
        registry_path=str(tmp_path / "registry.json"),
    )

    assert status["status"] == "PREDICTED"
    assert status["prediction_rows"] == 1
    output = pd.read_csv(tmp_path / "predictions.csv")
    assert output.loc[0, "prediction"] == "H"
    assert output.loc[0, "mom_status"] == "BLOCKED_UPSTREAM_PLAYER_MODEL"
    assert pd.isna(output.loc[0, "mom_1_player_id"])
    assert np.isnan(output.loc[0, "mom_1_probability"])
    assert output.loc[0, "market_over_2_5"] == 0.35
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "PREDICTED"
