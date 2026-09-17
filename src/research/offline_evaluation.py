"""Leakage-resistant offline evaluation for candidate external features.

The evaluator accepts precomputed predictions and labels only.  It does not
know or mutate the incumbent production model.  Rows are sorted by kickoff
order and are rejected when PIT provenance is missing or when the feature was
available after the prediction cutoff.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class EvaluationRow:
    kickoff_at: str
    prediction_cutoff_at: str
    label: int
    probabilities: tuple[float, ...]
    pit_safe: bool
    source_snapshot_id: str


@dataclass(frozen=True)
class EvaluationResult:
    n: int
    accuracy: float
    log_loss: float
    brier: float
    pit_safe_rows: int
    rejected_rows: int


def _time(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _validate_probabilities(p: Sequence[float], classes: int) -> tuple[float, ...]:
    if len(p) != classes:
        raise ValueError("probability vector length does not match class count")
    if any(not math.isfinite(x) or x < 0 or x > 1 for x in p):
        raise ValueError("probabilities must be finite and within [0, 1]")
    total = sum(p)
    if total <= 0 or abs(total - 1.0) > 1e-6:
        raise ValueError("1X2 probability vector must sum to 1")
    return tuple(float(x) for x in p)


def evaluate(rows: Iterable[EvaluationRow], *, classes: int = 3) -> EvaluationResult:
    accepted: list[EvaluationRow] = []
    rejected = 0
    ordered = sorted(rows, key=lambda r: _time(r.kickoff_at))
    previous = None
    for row in ordered:
        cutoff = _time(row.prediction_cutoff_at)
        kickoff = _time(row.kickoff_at)
        if cutoff > kickoff or not row.pit_safe:
            rejected += 1
            continue
        _validate_probabilities(row.probabilities, classes)
        if previous is not None and kickoff < previous:
            raise AssertionError("evaluation rows are not chronological")
        previous = kickoff
        accepted.append(row)

    if not accepted:
        raise ValueError("no PIT-safe rows available for offline evaluation")

    hits = sum(max(range(classes), key=lambda i: r.probabilities[i]) == r.label for r in accepted)
    ll = 0.0
    brier = 0.0
    for row in accepted:
        p = max(min(row.probabilities[row.label], 1 - 1e-15), 1e-15)
        ll -= math.log(p)
        brier += sum((row.probabilities[i] - (1.0 if i == row.label else 0.0)) ** 2 for i in range(classes))

    n = len(accepted)
    return EvaluationResult(n, hits / n, ll / n, brier / n, n, rejected)


def compare_candidate_to_baseline(
    baseline: Mapping[str, float], candidate: Mapping[str, float],
    *, metrics: Sequence[str] = ("accuracy", "log_loss", "brier"),
) -> dict[str, float]:
    """Return candidate-minus-baseline deltas; no winner/adoption decision is made."""
    return {metric: float(candidate[metric]) - float(baseline[metric]) for metric in metrics}
