"""Point-in-time-safe Man of the Match candidate selector.

MOM is a PLAYER prediction, not a match-result class.  The selector ranks
pre-match eligible players and returns exactly four candidates.  It must only
be fed features that were available before kickoff; post-match ratings,
MOTM labels and in-match statistics are targets/evidence, never features.
"""
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
    """Return exactly top_k MOM candidates, ranked by pre-match probability.

    The caller is responsible for producing probabilities from PIT-safe
    pre-match features. This function performs no label-derived feature work.
    """
    if top_k != 4:
        raise ValueError("MOM production output is fixed to exactly 4 candidates")
    ids = [str(x) for x in player_ids]
    probs = np.asarray(list(probabilities), dtype=float)
    if len(ids) != len(probs):
        raise ValueError("player_ids and probabilities must have equal length")
    if len(ids) < top_k:
        raise ValueError("at least 4 eligible players are required")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0):
        raise ValueError("probabilities must be finite and non-negative")
    order = np.argsort(-probs, kind="stable")[:top_k]
    selected = probs[order]
    total = float(selected.sum())
    if total <= 0:
        raise ValueError("top-4 probabilities must have positive mass")
    selected = selected / total
    return [
        MOMCandidate(ids[int(i)], float(p), rank)
        for rank, (i, p) in enumerate(zip(order, selected), start=1)
    ]
