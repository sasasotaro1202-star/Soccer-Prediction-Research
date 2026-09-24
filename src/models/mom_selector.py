"""Point-in-time-safe Man of the Match candidate selector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MOMCandidate:
    player_id: str
    probability: float
    rank: int


def select_mom_candidates(
    player_ids: Iterable[str],
    probabilities: Iterable[float],
    top_k: int = 4,
) -> list[MOMCandidate]:
    """Return exactly four MOM candidates with raw model probabilities.

    MOM probabilities are unconditional probabilities over the full eligible
    player set. They are deliberately *not* renormalized over the displayed
    top-4 candidates, so the displayed values remain calibrated probabilities
    and do not misleadingly sum to 100%.
    """
    if top_k != 4:
        raise ValueError("MOM production output is fixed to exactly 4 candidates")
    ids = [str(x) for x in player_ids]
    probs = np.asarray(list(probabilities), dtype=float)
    if len(ids) != len(probs):
        raise ValueError("player_ids and probabilities must have equal length")
    if len(set(ids)) != len(ids):
        raise ValueError("player_ids must be unique within a fixture")
    if len(ids) < top_k:
        raise ValueError("at least 4 eligible players are required")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0):
        raise ValueError("probabilities must be finite and non-negative")
    order = np.argsort(-probs, kind="stable")[:top_k]
    return [
        MOMCandidate(ids[int(i)], float(probs[int(i)]), rank)
        for rank, i in enumerate(order, start=1)
    ]
