import numpy as np
import pandas as pd

from src.research.mom_proxy import (
    FEATURES,
    PlayerHistory,
    _candidate_pool,
    _player_features,
    _proxy_winner,
    _softmax,
    run_mom_proxy_research,
)
from collections import deque


def test_proxy_label_uses_highest_positive_minutes_rating_with_deterministic_tie_break():
    frame = pd.DataFrame([
        {"player_id": "9", "player_name": "Nine", "minutes": 90, "rating": 7.8},
        {"player_id": "2", "player_name": "Two", "minutes": 88, "rating": 8.1},
        {"player_id": "1", "player_name": "One", "minutes": 70, "rating": 8.1},
        {"player_id": "8", "player_name": "Eight", "minutes": 0, "rating": 9.9},
    ])
    winner, name = _proxy_winner(frame)
    assert winner == "1"
    assert name == "One"


def test_player_features_are_built_from_history_only():
    state = PlayerHistory(deque([
        {
            "player_id": "p1",
            "team_id": "10",
            "kickoff_utc": pd.Timestamp("2025-01-01T12:00:00Z"),
            "rating": 7.0,
            "minutes": 90,
            "starter": 1.0,
        },
        {
            "player_id": "p1",
            "team_id": "10",
            "kickoff_utc": pd.Timestamp("2025-01-05T12:00:00Z"),
            "rating": 9.0,
            "minutes": 60,
            "starter": 0.0,
        },
    ], maxlen=20))
    values = _player_features(state, target_time=pd.Timestamp("2025-01-06T12:00:00Z"))
    assert values["apps_5"] == 2
    assert values["rating_mean_5"] == 8.0
    assert values["minutes_mean_5"] == 75.0
    assert values["last_rating"] == 9.0
    assert values["days_since_last_app"] == 1.0
    assert set(values) == set(FEATURES)


def test_candidate_pool_excludes_stale_transfer_and_old_player():
    states = {
        "p1": PlayerHistory(deque([{
            "team_id": "10",
            "kickoff_utc": pd.Timestamp("2025-01-01T12:00:00Z"),
            "rating": 8.0,
            "minutes": 90,
            "starter": 1.0,
        }], maxlen=20)),
        "p2": PlayerHistory(deque([{
            "team_id": "11",
            "kickoff_utc": pd.Timestamp("2025-01-01T12:00:00Z"),
            "rating": 9.0,
            "minutes": 90,
            "starter": 1.0,
        }], maxlen=20)),
        "p3": PlayerHistory(deque([
            {
                "team_id": "10",
                "kickoff_utc": pd.Timestamp("2025-01-01T12:00:00Z"),
                "rating": 7.0,
                "minutes": 90,
                "starter": 1.0,
            },
            {
                "team_id": "10",
                "kickoff_utc": pd.Timestamp("2026-01-01T12:00:00Z"),
                "rating": 7.5,
                "minutes": 90,
                "starter": 1.0,
            },
        ], maxlen=20)),
    }
    # Add p1/p3 to team 10, then verify latest-team identity is respected.
    pools = {"10": {"p1", "p2", "p3"}}
    result = _candidate_pool(
        "10",
        pools,
        states,
        target_time=pd.Timestamp("2026-02-01T12:00:00Z"),
    )
    assert "p2" not in result
    assert result == ["p3"]


def test_softmax_probability_mass_is_one():
    probs = _softmax([-1.0, 0.0, 2.0, 0.5])
    assert np.isfinite(probs).all()
    assert np.isclose(probs.sum(), 1.0, atol=1e-12)
    assert np.all(probs > 0.0)


def test_chronological_oos_research_keeps_exact_top4_contract():
    rows = []
    for match_index in range(160):
        winner = f"p{match_index % 4}"
        kickoff = pd.Timestamp("2020-01-01T12:00:00Z") + pd.Timedelta(days=match_index)
        for p in range(4):
            pid = f"p{p}"
            signal = 8.5 if pid == winner else 6.0 + 0.1 * p
            rows.append({
                "match_id": str(match_index),
                "competition": "EPL" if match_index % 2 == 0 else "J1",
                "kickoff_utc": kickoff,
                "player_id": pid,
                "player_name": pid,
                "proxy_winner_player_id": winner,
                **{
                    feature: (
                        signal if feature in {"rating_mean_5", "rating_ewm_5", "last_rating"}
                        else 1.0
                    )
                    for feature in FEATURES
                },
            })
    cases = pd.DataFrame(rows)
    result = run_mom_proxy_research(
        cases,
        min_train_fixtures=100,
        oos_block_fixtures=20,
        min_blocks=3,
    )
    state = result["state"]
    assert state["production_usable"] is False
    assert state["locked_oos_tuning"] is False
    output = pd.DataFrame(result["rows"])
    counts = output.groupby("match_id").size()
    assert counts.eq(4).all()
    mass = output.groupby(["model", "match_id"])["probability"].sum()
    assert np.isfinite(mass).all()
    assert (mass <= 1.0 + 1e-12).all()
