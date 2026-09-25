import numpy as np
import pandas as pd

from src.monitoring.dynamic_routing import (
    dynamic_route_weights,
    history_support_risk,
    routing_risk_score,
)


def test_history_support_risk_is_high_when_support_is_below_full_history():
    frame = pd.DataFrame({
        "home_history_support_n": [0, 5, 20, 30],
        "away_history_support_n": [2, 10, 25, 40],
    })
    risk = history_support_risk(frame)
    # The risk is governed by the weaker team's support and linearly decays
    # from 1.0 at min_games=5 to 0.0 at full_games=20.
    assert np.isclose(risk[0], 1.0)
    assert np.isclose(risk[1], 1.0)
    assert np.isclose(risk[2], 0.0)
    assert np.isclose(risk[3], 0.0)
    assert np.all(np.diff(risk) <= 0.0)


def test_dynamic_routing_uses_sparse_support_conservatively():
    base = np.asarray([[0.9, 0.1], [0.9, 0.1]], dtype=float)
    fallback = np.asarray([0.5, 0.5], dtype=float)
    probs = {
        "a": np.asarray([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]], dtype=float),
        "b": np.asarray([[0.7, 0.2, 0.1], [0.7, 0.2, 0.1]], dtype=float),
    }
    weights, diag = dynamic_route_weights(
        base, fallback, probs, np.asarray([0.0, 0.0]),
        drift_strength=0.0, uncertainty_strength=0.0, min_specialist_trust=0.05,
        support_scores=np.asarray([0.0, 1.0]), support_strength=1.0,
    )
    assert diag["support"][1] > diag["support"][0]
    assert diag["trust"][1] < diag["trust"][0]
    assert abs(weights[1, 0] - fallback[0]) < abs(weights[0, 0] - fallback[0])


def test_routing_risk_support_component_is_bounded():
    risk = routing_risk_score(
        np.asarray([0.0, 0.5]), np.asarray([0.0, 0.2]),
        support_scores=np.asarray([0.0, 1.0]), support_weight=0.15,
    )
    assert np.all((risk >= 0.0) & (risk <= 1.0))
    assert risk[1] > risk[0]