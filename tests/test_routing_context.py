import numpy as np
import pandas as pd

from src.evaluation.walk_forward import _routing_context


def test_routing_context_is_prediction_time_only_and_deterministic():
    frame = pd.DataFrame([
        {
            "competition": "EPL",
            "elo_diff": 220.0,
            "home_goal_total_avg_5": 3.0,
            "away_goal_total_avg_5": 2.8,
            "rest_diff_hours": 30.0,
            "neutral_venue_known": True,
            "neutral_venue": False,
        },
        {
            "competition": "AG_M",
            "elo_diff": -90.0,
            "rest_diff_hours": -10.0,
            "neutral_venue_known": False,
            "neutral_venue": pd.NA,
        },
    ])
    a = _routing_context(frame)
    b = _routing_context(frame)
    assert a["routing_context"].tolist() == b["routing_context"].tolist()
    assert "EPL|LARGE_HOME|HIGH|MISSING|HOME_MAJOR|HOME_AWAY" == a.loc[0, "routing_context"]
    assert "AG_M|AWAY|MISSING|MISSING|AWAY_SMALL|UNKNOWN" == a.loc[1, "routing_context"]


def test_routing_context_handles_missing_numeric_columns():
    frame = pd.DataFrame([{"competition": "EPL"}])
    out = _routing_context(frame)
    assert out.loc[0, "routing_strength_gap"] == "MISSING"
    assert out.loc[0, "routing_scoring_environment"] == "MISSING"
    assert out.loc[0, "routing_draw_environment"] == "MISSING"
    assert out.loc[0, "routing_rest"] == "MISSING"
    assert out.loc[0, "routing_venue"] == "UNKNOWN"
    assert isinstance(out.loc[0, "routing_context"], str)
