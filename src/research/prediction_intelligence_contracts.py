"""Research-only contracts for forecast lifetime, information acquisition, and counterfactual failure analysis.

This module is deliberately side-effect free. It does not fetch data, mutate the
Production registry, change probabilities, or authorize model promotion. The
contracts are scaffolding for later PIT-safe/OOS-validated research lanes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Iterable, Sequence


class ForecastLifetime(str, Enum):
    FRESH = "FRESH"
    AGING = "AGING"
    STALE = "STALE"
    INVALIDATED = "INVALIDATED"


@dataclass(frozen=True)
class ForecastLifetimePolicy:
    """Policy for deciding when a forecast becomes operationally stale."""

    ttl_seconds: int
    aging_fraction: float = 0.50

    def __post_init__(self) -> None:
        if self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        if not 0.0 < self.aging_fraction < 1.0:
            raise ValueError("aging_fraction must be between 0 and 1")


@dataclass(frozen=True)
class ForecastLifetimeState:
    state: ForecastLifetime
    age_seconds: float
    reason: str
    production_usable: bool = False


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def assess_forecast_lifetime(
    *,
    prediction_cutoff: datetime | None,
    evaluated_at: datetime,
    policy: ForecastLifetimePolicy,
    invalidation_reason: str | None = None,
) -> ForecastLifetimeState:
    """Assess lifetime conservatively; missing/invalid timing fails closed.

    This function is intentionally independent of prediction probabilities.
    """
    evaluated_at = _utc(evaluated_at)
    if prediction_cutoff is None:
        return ForecastLifetimeState(
            ForecastLifetime.INVALIDATED,
            math.nan,
            "MISSING_PREDICTION_CUTOFF",
            False,
        )

    cutoff = _utc(prediction_cutoff)
    age_seconds = (evaluated_at - cutoff).total_seconds()
    if age_seconds < 0:
        return ForecastLifetimeState(
            ForecastLifetime.INVALIDATED,
            age_seconds,
            "CLOCK_SKEW_EVALUATION_BEFORE_CUTOFF",
            False,
        )
    if invalidation_reason:
        return ForecastLifetimeState(
            ForecastLifetime.INVALIDATED,
            age_seconds,
            invalidation_reason,
            False,
        )
    if age_seconds >= policy.ttl_seconds:
        return ForecastLifetimeState(
            ForecastLifetime.STALE,
            age_seconds,
            "TTL_EXCEEDED",
            False,
        )
    if age_seconds >= policy.ttl_seconds * policy.aging_fraction:
        return ForecastLifetimeState(
            ForecastLifetime.AGING,
            age_seconds,
            "AGING_THRESHOLD_REACHED",
            False,
        )
    return ForecastLifetimeState(
        ForecastLifetime.FRESH,
        age_seconds,
        "WITHIN_TTL",
        False,
    )


class AcquisitionAction(str, Enum):
    PREDICT_NOW = "PREDICT_NOW"
    ACQUIRE_MORE = "ACQUIRE_MORE"
    WAIT = "WAIT"
    RECOMPUTE = "RECOMPUTE"
    ROUTE = "ROUTE"
    FALLBACK = "FALLBACK"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class InformationAcquisitionOption:
    """Normalized research option; costs are deliberately unitless [0,1]."""

    name: str
    action: AcquisitionAction
    expected_information_gain: float
    expected_error_reduction: float
    latency_cost: float
    cost_score: float
    failure_risk: float
    pit_verified: bool

    def __post_init__(self) -> None:
        for field_name in (
            "expected_information_gain",
            "expected_error_reduction",
            "latency_cost",
            "cost_score",
            "failure_risk",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be within [0,1]")


@dataclass(frozen=True)
class InformationAcquisitionPolicy:
    information_weight: float = 0.30
    error_reduction_weight: float = 0.40
    latency_weight: float = 0.10
    cost_weight: float = 0.10
    failure_risk_weight: float = 0.10
    max_latency_cost: float = 1.0
    max_cost_score: float = 1.0
    max_failure_risk: float = 1.0

    def __post_init__(self) -> None:
        weights = (
            self.information_weight,
            self.error_reduction_weight,
            self.latency_weight,
            self.cost_weight,
            self.failure_risk_weight,
        )
        if any(w < 0.0 or not math.isfinite(w) for w in weights):
            raise ValueError("policy weights must be finite and non-negative")
        if sum(weights) <= 0.0:
            raise ValueError("at least one policy weight must be positive")
        for field_name in ("max_latency_cost", "max_cost_score", "max_failure_risk"):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be within [0,1]")


@dataclass(frozen=True)
class RankedAcquisitionOption:
    option: InformationAcquisitionOption
    utility_score: float
    admissible: bool
    reason: str


def rank_information_acquisition(
    options: Sequence[InformationAcquisitionOption],
    *,
    policy: InformationAcquisitionPolicy = InformationAcquisitionPolicy(),
) -> tuple[RankedAcquisitionOption, ...]:
    """Rank options while fail-closing on PIT and operational risk.

    Unknown PIT is never converted into a positive utility.
    """
    ranked: list[RankedAcquisitionOption] = []
    total_weight = (
        policy.information_weight
        + policy.error_reduction_weight
        + policy.latency_weight
        + policy.cost_weight
        + policy.failure_risk_weight
    )
    for option in options:
        reasons: list[str] = []
        if not option.pit_verified:
            reasons.append("PIT_UNVERIFIED")
        if option.latency_cost > policy.max_latency_cost:
            reasons.append("LATENCY_LIMIT")
        if option.cost_score > policy.max_cost_score:
            reasons.append("COST_LIMIT")
        if option.failure_risk > policy.max_failure_risk:
            reasons.append("FAILURE_RISK_LIMIT")

        raw = (
            policy.information_weight * option.expected_information_gain
            + policy.error_reduction_weight * option.expected_error_reduction
            - policy.latency_weight * option.latency_cost
            - policy.cost_weight * option.cost_score
            - policy.failure_risk_weight * option.failure_risk
        ) / total_weight
        score = float(raw) if not reasons else float("-inf")
        ranked.append(
            RankedAcquisitionOption(
                option=option,
                utility_score=score,
                admissible=not reasons,
                reason=";".join(reasons) if reasons else "ADMISSIBLE",
            )
        )

    return tuple(
        sorted(
            ranked,
            key=lambda item: (
                not item.admissible,
                -item.utility_score if item.admissible else 0.0,
                item.option.name,
            ),
        )
    )


@dataclass(frozen=True)
class CounterfactualFact:
    name: str
    observed_value: str
    source: str


@dataclass(frozen=True)
class CounterfactualHypothesis:
    intervention: str
    hypothesis: str
    evidence_level: str = "HYPOTHESIS"
    tested: bool = False

    def __post_init__(self) -> None:
        if self.evidence_level != "HYPOTHESIS":
            raise ValueError("counterfactual candidates must remain explicitly labelled HYPOTHESIS")


@dataclass(frozen=True)
class CounterfactualFailureRecord:
    failure_id: str
    event_id: str
    target: str
    failure_type: str
    measured_facts: tuple[CounterfactualFact, ...]
    hypotheses: tuple[CounterfactualHypothesis, ...]
    production_changed: bool = False
    frozen_holdout_touched: bool = False


def build_counterfactual_failure_record(
    *,
    failure_id: str,
    event_id: str,
    target: str,
    failure_type: str,
    measured_facts: Iterable[CounterfactualFact],
    hypotheses: Iterable[CounterfactualHypothesis],
) -> CounterfactualFailureRecord:
    """Preserve measured facts and rescue hypotheses as different evidence types."""
    facts = tuple(measured_facts)
    ideas = tuple(hypotheses)
    if any(not fact.name or not fact.source for fact in facts):
        raise ValueError("measured facts require name and source")
    if any(not idea.intervention or not idea.hypothesis for idea in ideas):
        raise ValueError("counterfactual hypotheses require intervention and hypothesis")
    return CounterfactualFailureRecord(
        failure_id=failure_id,
        event_id=event_id,
        target=target,
        failure_type=failure_type,
        measured_facts=facts,
        hypotheses=ideas,
        production_changed=False,
        frozen_holdout_touched=False,
    )
