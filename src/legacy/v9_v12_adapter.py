"""Compatibility boundary for the legacy Soccer V9/V12 engine.

The legacy implementation remains the baseline. The Research Engine owns PIT,
chronological OOS, comparison, and adoption decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class LegacyPrediction:
    match_id: str
    prediction_cutoff_at_utc: str
    probabilities: Mapping[str, float]
    score_candidates: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_version: str = "V9/V12"


@dataclass(frozen=True)
class LegacyEngineAdapter:
    """Dependency-injected bridge around existing V9/V12 callables."""

    predict_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    source_version: str = "V9/V12"

    def predict(self, context: Mapping[str, Any]) -> LegacyPrediction:
        cutoff = context.get("prediction_cutoff_at_utc")
        match_id = context.get("match_id")
        if not cutoff:
            raise ValueError("prediction_cutoff_at_utc is required")
        if not match_id:
            raise ValueError("match_id is required")
        if context.get("pit_verified") is not True:
            raise ValueError("Legacy prediction requires PIT-verified context")

        raw = dict(self.predict_fn(context))
        probabilities = raw.get("probabilities")
        if not isinstance(probabilities, Mapping):
            raise ValueError("Legacy predict_fn must return 'probabilities'")
        required = {"H", "D", "A"}
        if not required.issubset(probabilities):
            raise ValueError("Legacy probabilities must contain H, D, and A")
        total = sum(float(probabilities[k]) for k in required)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Legacy probabilities must sum to 1; got {total}")
        return LegacyPrediction(
            match_id=str(match_id),
            prediction_cutoff_at_utc=str(cutoff),
            probabilities={k: float(probabilities[k]) for k in required},
            score_candidates=tuple(raw.get("score_candidates", ())),
            metadata=dict(raw.get("metadata", {})),
            source_version=self.source_version,
        )


def make_pit_context(record: Mapping[str, Any]) -> dict[str, Any]:
    required = ("match_id", "prediction_cutoff_at_utc", "pit_verified")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"Missing adapter context fields: {missing}")
    if record["pit_verified"] is not True:
        raise ValueError("Cannot adapt a non-PIT-verified record")
    return dict(record)
