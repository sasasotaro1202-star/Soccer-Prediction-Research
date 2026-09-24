import numpy as np
import pandas as pd

from src.monitoring.dynamic_routing import build_drift_reference, compute_drift_scores, dynamic_route_weights, normalized_entropy, routing_risk_bucket, routing_risk_score


def test_drift_reference_is_deterministic_and_reference_only():
    ref = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [10.0, 11.0, 12.0, 13.0]})
    assert build_drift_reference(ref, ["x", "y"]) == build_drift_reference(ref, ["x", "y"])


def test_shifted_rows_have_more_drift_than_reference_rows():
    ref = pd.DataFrame({"x": np.linspace(0, 1, 100)})
    reference = build_drift_reference(ref, ["x"])
    current = pd.DataFrame({"x": [0.5, 0.51, 0.49, 0.50]})
    shifted = pd.DataFrame({"x": [10.0, 10.1, 9.9, 10.2]})
    assert compute_drift_scores(current, reference, ["x"]).mean() < compute_drift_scores(shifted, reference, ["x"]).mean()


def test_dynamic_routing_moves_toward_global_on_drift_and_uncertainty():
    base = np.asarray([[0.9, 0.1], [0.9, 0.1]], dtype=float)
    fallback = np.asarray([0.5, 0.5], dtype=float)
    probs = {
        "a": np.asarray([[0.49, 0.33, 0.18], [1 / 3, 1 / 3, 1 / 3]], dtype=float),
        "b": np.asarray([[0.51, 0.34, 0.15], [1 / 3, 1 / 3, 1 / 3]], dtype=float),
    }
    weights, diag = dynamic_route_weights(base, fallback, probs, np.asarray([0.0, 1.0]), drift_strength=1.0, uncertainty_strength=1.0, min_specialist_trust=0.10)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert diag["trust"][1] < diag["trust"][0]
    assert abs(weights[1, 0] - fallback[0]) < abs(weights[0, 0] - fallback[0])


def test_entropy_is_bounded():
    low = normalized_entropy(np.asarray([[0.98, 0.01, 0.01]]))[0]
    high = normalized_entropy(np.asarray([[1 / 3, 1 / 3, 1 / 3]]))[0]
    assert low < high
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0


def test_routing_risk_score_and_bucket_are_bounded():
    scores = routing_risk_score(np.asarray([0.0, 0.5, 1.0]), np.asarray([0.0, 0.2, 1.0]))
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    buckets = routing_risk_bucket(scores)
    assert list(buckets) == ["LOW", "MEDIUM", "HIGH"]
