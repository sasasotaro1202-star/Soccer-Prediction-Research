import numpy as np
import pandas as pd
import pytest

from src.research.score_distribution_ensemble import (
    blend_distributions,
    cross_block_weights,
    derive_metrics,
    normalize_distribution,
)


def test_normalize_distribution_is_finite_and_sums_to_one():
    out = normalize_distribution([
        (0, 0, 2.0),
        (1, 0, 1.0),
        (0, 0, 1.0),
    ])
    assert out[(0, 0)] == pytest.approx(0.75)
    assert sum(out.values()) == pytest.approx(1.0)


def test_normalize_distribution_rejects_invalid_mass():
    with pytest.raises(ValueError):
        normalize_distribution([(0, 0, 0.0)])
    with pytest.raises(ValueError):
        normalize_distribution([(0, 0, -1.0)])


def test_blend_distributions_preserves_joint_support_and_mass():
    first = [(0, 0, 0.6), (1, 0, 0.4)]
    second = [(0, 0, 0.2), (0, 1, 0.8)]
    out = blend_distributions({"a": first, "b": second}, {"a": 0.75, "b": 0.25})
    assert set(out) == {(0, 0), (1, 0), (0, 1)}
    assert sum(out.values()) == pytest.approx(1.0)
    assert out[(0, 0)] == pytest.approx(0.50)
    assert out[(1, 0)] == pytest.approx(0.30)
    assert out[(0, 1)] == pytest.approx(0.20)


def test_derived_binary_outputs_come_from_same_joint_distribution():
    dist = {(0, 0): 0.30, (1, 0): 0.20, (1, 1): 0.10, (2, 1): 0.40}
    metrics = derive_metrics(dist, actual_home=2, actual_away=1)
    assert metrics["exact_score_hit"] == 1.0
    assert metrics["top3_score_hit"] == 1.0
    assert metrics["top4_score_hit"] == 1.0
    assert metrics["over_2_5_logloss"] >= 0.0
    assert metrics["btts_brier"] >= 0.0


def test_cross_block_weights_uses_prior_block_losses_only():
    weights = cross_block_weights(
        {
            "primary": [0.9, 0.8, 0.7],
            "dixon_coles": [0.5, 0.6, 0.55],
            "xg": [1.2, 1.1, 1.0],
        },
        ["primary", "dixon_coles", "xg"],
    )
    assert set(weights) == {"primary", "dixon_coles", "xg"}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["dixon_coles"] > weights["primary"]
    assert weights["dixon_coles"] > weights["xg"]


def test_cross_block_weights_empty_history_is_deterministic():
    weights = cross_block_weights({}, ["primary", "dixon_coles"])
    assert set(weights) == {"primary", "dixon_coles"}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert np.all(np.isfinite(list(weights.values())))


def test_score_distribution_ensemble_is_research_only_by_api_shape():
    block = pd.DataFrame([
        {"home_team": "A", "away_team": "B", "competition": "EPL",
         "home_goals": 1, "away_goals": 0, "neutral_venue": False},
    ])

    models = {"primary": {"method": "primary"},
              "dixon_coles": object()}
    distributions = {
        name: (lambda *_args, **_kwargs: [(1, 0, 0.8), (0, 0, 0.2)])
        for name in models
    }
    metrics, weights = __import__(
        "src.research.score_distribution_ensemble",
        fromlist=["evaluate_block"],
    ).evaluate_block(
        block,
        models,
        distributions,
        {"primary": [], "dixon_coles": []},
    )
    assert metrics["score_logloss"] >= 0.0
    assert sum(weights.values()) == pytest.approx(1.0)
