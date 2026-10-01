from __future__ import annotations

import numpy as np
import pytest

from src.research.prediction_method_meta_policy import (
    POLICY_VERSION,
    select_method,
    validate_decision,
)


def test_high_confidence_is_primary_without_instinctively_changing_probability():
    p = [0.92, 0.05, 0.03]
    result = select_method(
        probabilities=p,
        strategy="validation_optimized_contextual_ensemble",
        router_status="GLOBAL",
        pit_safe=True,
        raw_predictability=0.90,
        score_distribution_available=True,
        mom_available=False,
        freshness_score=0.90,
    )
    assert result["status"] == "SELECTED"
    assert result["output"]["action"] == "PRIMARY"
    assert result["output"]["format_candidate"] == "probability_distribution"
    assert result["probability"]["home"] == pytest.approx(p[0])
    assert result["probability"]["draw"] == pytest.approx(p[1])
    assert result["probability"]["away"] == pytest.approx(p[2])
    assert result["probability"]["probability_changed"] is False


def test_high_confidence_low_predictability_triggers_review():
    result = select_method(
        probabilities=[0.92, 0.05, 0.03],
        strategy="validation_optimized_contextual_ensemble",
        router_status="FULL:example",
        pit_safe=True,
        raw_predictability=0.20,
        model_disagreement=0.07,
        risk_score=0.75,
        score_distribution_available=True,
        freshness_score=0.90,
    )
    assert result["uncertainty"]["level"] == "HIGH"
    assert result["uncertainty"]["high_confidence_low_predictability"] is True
    assert result["output"]["action"] == "REVIEW"
    assert result["output"]["format_candidate"] == "prediction_set_or_scenario"
    assert result["information"]["next_action"] == "recompute_candidate"


def test_missing_raw_predictability_can_be_derived_from_outcome_free_telemetry():
    result = select_method(
        probabilities=[0.68, 0.18, 0.14],
        strategy="fixed_equal_weight",
        router_status="GLOBAL",
        pit_safe=True,
        model_disagreement=0.02,
        predictive_entropy=0.30,
        risk_score=0.20,
        score_distribution_available=False,
        freshness_score=0.90,
    )
    assert result["uncertainty"]["predictability"] is not None
    assert 0.0 <= result["uncertainty"]["predictability"] <= 1.0
    assert result["uncertainty"]["predictability_derived"] is True


def test_stale_information_requests_refresh_without_rewriting_probability():
    result = select_method(
        probabilities=[0.70, 0.20, 0.10],
        strategy="fixed_equal_weight",
        router_status="GLOBAL",
        pit_safe=True,
        raw_predictability=0.80,
        freshness_score=0.20,
    )
    assert result["information"]["next_action"] == "refresh_pit_safe_sources"
    assert result["probability"]["probability_changed"] is False


def test_fail_closed_when_pit_is_not_safe():
    result = select_method(
        probabilities=[0.60, 0.25, 0.15],
        strategy="fixed_equal_weight",
        router_status="GLOBAL",
        pit_safe=False,
    )
    assert result["status"] == "FAIL_CLOSED"
    assert result["output"]["action"] == "ABSTAIN"
    assert result["output"]["format_candidate"] == "abstain_or_fallback"
    assert result["uncertainty"]["predictability"] is None


def test_probability_input_is_normalized_but_not_policy_shifted():
    result = select_method(
        probabilities=[60, 30, 10],
        strategy="fixed_equal_weight",
        router_status="GLOBAL",
        pit_safe=True,
        raw_predictability=0.70,
    )
    assert np.isclose(
        [
            result["probability"]["home"],
            result["probability"]["draw"],
            result["probability"]["away"],
        ],
        [0.6, 0.3, 0.1],
    ).all()
    assert result["probability"]["probability_changed"] is False


def test_policy_validation_is_fail_closed_for_invalid_promotion_flag():
    result = select_method(
        probabilities=[0.60, 0.25, 0.15],
        strategy="fixed_equal_weight",
        router_status="GLOBAL",
        pit_safe=True,
        raw_predictability=0.70,
    )
    validate_decision(result)
    bad = dict(result)
    bad["safety"] = dict(result["safety"])
    bad["safety"]["production_auto_promotion"] = True
    with pytest.raises(ValueError):
        validate_decision(bad)


def test_policy_version_and_hash_are_present():
    result = select_method(
        probabilities=[0.60, 0.25, 0.15],
        strategy="fixed_equal_weight",
        router_status="GLOBAL",
        pit_safe=True,
        raw_predictability=0.70,
    )
    assert result["policy_version"] == POLICY_VERSION
    assert len(result["policy_hash"]) == 64
    validate_decision(result)
