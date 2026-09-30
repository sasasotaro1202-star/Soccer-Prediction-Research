from __future__ import annotations

import pandas as pd

from src.prediction.target_competition_policy import annotate_predictions, build_target_policy


def test_target_policy_is_explicit_for_each_requested_target():
    row = pd.Series({
        "competition": "EPL",
        "elo_diff": 25.0,
        "home_goal_total_avg_5": 2.2,
        "away_goal_total_avg_5": 2.0,
        "home_draw_rate_20": 0.25,
        "away_draw_rate_20": 0.27,
        "rest_diff_hours": 3.0,
        "neutral_venue_known": True,
        "neutral_venue": False,
    })
    bundle = {
        "weights": {"logistic": 0.5, "extra_trees": 0.5},
        "score_method": "dixon_coles",
        "score_model": {"competition_rates": {"EPL": {"home_mean": 1.5, "away_mean": 1.2}}},
        "routing_policy": {
            "fallback_weights": {"logistic": 0.5, "extra_trees": 0.5},
            "context_weights": {
                "COMP:EPL": {"logistic": 0.6, "extra_trees": 0.4},
            },
            "dynamic_routing": {"enabled": True},
        },
        "score_method_by_competition": {
            "EPL": {"method": "dixon_coles", "status": "COMPETITION_SPECIALIST"},
        },
    }
    p = build_target_policy(row, bundle)
    assert p["competition"] == "EPL"
    assert p["target_1x2_policy"].startswith("1X2|")
    assert p["target_score_policy"] == "Score|dixon_coles|COMPETITION_SPECIALIST|COMPETITION_RATE"
    assert p["target_ou_policy"] == "O/U|DERIVED_SCORE_DISTRIBUTION|dixon_coles|COMPETITION_SPECIALIST|COMPETITION_RATE"
    assert p["target_btts_policy"] == "BTTS|DERIVED_SCORE_DISTRIBUTION|dixon_coles|COMPETITION_SPECIALIST|COMPETITION_RATE"
    assert p["target_mom_policy"] == "MOM|UPSTREAM_PLAYER_MODEL_REQUIRED"


def test_annotation_preserves_fixture_row_count_and_competition():
    fixtures = pd.DataFrame([
        {
            "match_id": "1",
            "competition": "EPL",
        },
        {
            "match_id": "2",
            "competition": "UCL",
        },
    ])
    bundle = {"weights": {"logistic": 1.0}, "score_method": "primary", "score_model": {"competition_rates": {}}}
    out = annotate_predictions(fixtures, bundle)
    assert len(out) == len(fixtures)
    assert out["policy_competition"].tolist() == ["EPL", "UCL"]
    assert out["policy_target_mom_policy"].eq("MOM|UPSTREAM_PLAYER_MODEL_REQUIRED").all()
