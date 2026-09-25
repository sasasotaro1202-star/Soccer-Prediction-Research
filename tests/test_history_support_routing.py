import numpy as np
import pandas as pd

from src.monitoring.dynamic_routing import history_support_risk, dynamic_route_weights


def test_history_support_risk_is_conservative():
    frame = pd.DataFrame({
        "home_history_support_n": [0, 5, 12, 20, 30],
        "away_history_support_n": [2, 10, 15, 25, 40],
    })
    risk = history_support_risk(frame)
    assert np.all((risk >= 0.0) & (risk <= 1.0))
    assert np.isclose(risk[0], 1.0)
    assert np.isclose(risk[1], 1.0)
    assert 0.0 < risk[2] < 1.0
    assert np.isclose(risk[3], 0.0)
    assert np.isclose(risk[4], 0.0)


def test_sparse_support_moves_contextual_weights_toward_fallback():
    base = np.asarray([[0.9, 0.1], [0.9, 0.1]], dtype=float)
    fallback = np.asarray([0.5, 0.5], dtype=float)
    probs = {
        "a": np.asarray([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]], dtype=float),
        "b": np.asarray([[0.7, 0.2, 0.1], [0.7, 0.2, 0.1]], dtype=float),
    }
    weights, diag = dynamic_route_weights(
        base, fallback, probs, np.zeros(2),
        drift_strength=0.0,
        uncertainty_strength=0.0,
        support_scores=np.asarray([0.0, 1.0]),
        support_strength=1.0,
    )
    assert diag["support"][1] > diag["support"][0]
    assert diag["trust"][1] < diag["trust"][0]
    assert abs(weights[1, 0] - fallback[0]) < abs(weights[0, 0] - fallback[0])
