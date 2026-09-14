"""Scoreline candidate selection for production output.

The score model may produce a full goal-grid probability distribution. Production
output is intentionally limited to the three highest-probability scorelines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ScoreCandidate:
    home_goals: int
    away_goals: int
    probability: float
    rank: int


def select_score_candidates(
    home_goals: Iterable[int],
    away_goals: Iterable[int],
    probabilities: Iterable[float],
    top_k: int = 3,
) -> list[ScoreCandidate]:
    if top_k != 3:
        raise ValueError("Score production output is fixed to exactly 3 candidates")
    hs = list(home_goals)
    aws = list(away_goals)
    probs = np.asarray(list(probabilities), dtype=float)
    if not (len(hs) == len(aws) == len(probs)):
        raise ValueError("score arrays must have equal length")
    if len(probs) < top_k:
        raise ValueError("at least 3 score candidates are required")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0):
        raise ValueError("probabilities must be finite and non-negative")
    order = np.argsort(-probs, kind="stable")[:top_k]
    selected = probs[order]
    total = float(selected.sum())
    if total <= 0:
        raise ValueError("top-3 probabilities must have positive mass")
    selected = selected / total
    return [
        ScoreCandidate(int(hs[int(i)]), int(aws[int(i)]), float(p), rank)
        for rank, (i, p) in enumerate(zip(order, selected), start=1)
    ]
