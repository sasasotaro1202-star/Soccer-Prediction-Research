from datetime import datetime, timedelta, timezone

import pytest

from src.research.prediction_intelligence_contracts import (
    AcquisitionAction,
    CounterfactualFact,
    CounterfactualHypothesis,
    ForecastLifetime,
    ForecastLifetimePolicy,
    InformationAcquisitionOption,
    InformationAcquisitionPolicy,
    assess_forecast_lifetime,
    build_counterfactual_failure_record,
    rank_information_acquisition,
)


UTC = timezone.utc
BASE = datetime(2026, 10, 8, 6, 0, tzinfo=UTC)


def test_forecast_lifetime_is_fail_closed_and_time_aware():
    policy = ForecastLifetimePolicy(ttl_seconds=3600, aging_fraction=0.5)

    assert assess_forecast_lifetime(
        prediction_cutoff=BASE,
        evaluated_at=BASE + timedelta(minutes=10),
        policy=policy,
    ).state is ForecastLifetime.FRESH

    assert assess_forecast_lifetime(
        prediction_cutoff=BASE,
        evaluated_at=BASE + timedelta(minutes=30),
        policy=policy,
    ).state is ForecastLifetime.AGING

    assert assess_forecast_lifetime(
        prediction_cutoff=BASE,
        evaluated_at=BASE + timedelta(hours=1),
        policy=policy,
    ).state is ForecastLifetime.STALE

    assert assess_forecast_lifetime(
        prediction_cutoff=None,
        evaluated_at=BASE,
        policy=policy,
    ).state is ForecastLifetime.INVALIDATED

    assert assess_forecast_lifetime(
        prediction_cutoff=BASE,
        evaluated_at=BASE + timedelta(minutes=5),
        policy=policy,
        invalidation_reason="LINEUP_UPDATE",
    ).state is ForecastLifetime.INVALIDATED


def test_forecast_lifetime_rejects_naive_timestamp():
    policy = ForecastLifetimePolicy(ttl_seconds=3600)
    with pytest.raises(ValueError, match="timezone-aware"):
        assess_forecast_lifetime(
            prediction_cutoff=datetime(2026, 10, 8, 6, 0),
            evaluated_at=BASE,
            policy=policy,
        )


def test_information_acquisition_prioritizes_admissible_high_value_option():
    options = (
        InformationAcquisitionOption(
            name="verified_team_news",
            action=AcquisitionAction.ACQUIRE_MORE,
            expected_information_gain=0.80,
            expected_error_reduction=0.70,
            latency_cost=0.20,
            cost_score=0.00,
            failure_risk=0.10,
            pit_verified=True,
        ),
        InformationAcquisitionOption(
            name="unknown_availability_source",
            action=AcquisitionAction.ACQUIRE_MORE,
            expected_information_gain=1.00,
            expected_error_reduction=1.00,
            latency_cost=0.00,
            cost_score=0.00,
            failure_risk=0.00,
            pit_verified=False,
        ),
        InformationAcquisitionOption(
            name="slow_source",
            action=AcquisitionAction.RECOMPUTE,
            expected_information_gain=0.95,
            expected_error_reduction=0.90,
            latency_cost=1.00,
            cost_score=0.10,
            failure_risk=0.05,
            pit_verified=True,
        ),
    )
    ranked = rank_information_acquisition(
        options,
        policy=InformationAcquisitionPolicy(max_latency_cost=0.90),
    )

    assert ranked[0].option.name == "verified_team_news"
    assert ranked[0].admissible is True
    assert any(
        item.option.name == "unknown_availability_source" and not item.admissible
        for item in ranked
    )
    assert ranked[-1].option.name == "unknown_availability_source"


def test_information_acquisition_rejects_unknown_pit_even_when_score_is_high():
    option = InformationAcquisitionOption(
        name="unknown",
        action=AcquisitionAction.ACQUIRE_MORE,
        expected_information_gain=1.0,
        expected_error_reduction=1.0,
        latency_cost=0.0,
        cost_score=0.0,
        failure_risk=0.0,
        pit_verified=False,
    )
    ranked = rank_information_acquisition([option])
    assert ranked[0].admissible is False
    assert ranked[0].utility_score == float("-inf")
    assert ranked[0].reason == "PIT_UNVERIFIED"


def test_counterfactual_failure_record_keeps_facts_and_hypotheses_separate():
    record = build_counterfactual_failure_record(
        failure_id="F-001",
        event_id="match-001",
        target="1X2",
        failure_type="MODEL_FAILURE",
        measured_facts=(
            CounterfactualFact(
                name="model_disagreement",
                observed_value="0.42",
                source="diagnostic snapshot",
            ),
        ),
        hypotheses=(
            CounterfactualHypothesis(
                intervention="acquire verified lineup",
                hypothesis="the additional context may have reduced uncertainty",
            ),
            CounterfactualHypothesis(
                intervention="route to specialist",
                hypothesis="the competition specialist may have reduced error",
            ),
        ),
    )

    assert record.measured_facts[0].source == "diagnostic snapshot"
    assert record.hypotheses[0].evidence_level == "HYPOTHESIS"
    assert record.production_changed is False
    assert record.frozen_holdout_touched is False


def test_counterfactual_rejects_measured_level_claims():
    with pytest.raises(ValueError, match="HYPOTHESIS"):
        CounterfactualHypothesis(
            intervention="alternate model",
            hypothesis="this would have won",
            evidence_level="MEASURED_FACT",
        )
