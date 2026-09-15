"""Select the top production Score candidates without inventing probability mass."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable


@dataclass(frozen=True)
class ScoreCandidate:
    home_goals: int
    away_goals: int
    probability: float
    rank: int
    display_probability: float


def select_score_candidates(
    candidates: Iterable[tuple[int, int, float]], top_k: int = 3
) -> list[ScoreCandidate]:
    """Return exactly ``top_k`` scorelines, preserving raw and display probabilities.

    ``probability`` is the unconditional probability from the full score
    distribution. ``display_probability`` is conditional on the selected
    top-k set and therefore sums to one. This distinction prevents the UI
    probability from being mistaken for the model's full-distribution mass.
    """
    if top_k != 3:
        raise ValueError("production Score output must contain exactly 3 candidates")

    rows = list(candidates)
    if len(rows) < top_k:
        raise ValueError("at least 3 score candidates are required")

    validated: list[tuple[int, int, float, int]] = []
    for index, (home_goals, away_goals, probability) in enumerate(rows):
        if not isinstance(home_goals, int) or home_goals < 0:
            raise ValueError("home_goals must be a non-negative integer")
        if not isinstance(away_goals, int) or away_goals < 0:
            raise ValueError("away_goals must be a non-negative integer")
        probability = float(probability)
        if not isfinite(probability) or probability < 0.0:
            raise ValueError("probability must be finite and non-negative")
        validated.append((home_goals, away_goals, probability, index))

    ranked = sorted(validated, key=lambda row: (-row[2], row[3]))[:top_k]
    selected_mass = sum(row[2] for row in ranked)
    if not isfinite(selected_mass) or selected_mass <= 0.0:
        raise ValueError("selected score candidates must contain positive probability mass")

    return [
        ScoreCandidate(
            home_goals=home_goals,
            away_goals=away_goals,
            probability=probability,
            rank=rank,
            display_probability=probability / selected_mass,
        )
        for rank, (home_goals, away_goals, probability, _index) in enumerate(ranked, start=1)
    ]
