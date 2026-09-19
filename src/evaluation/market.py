"""Market probability utilities used only when odds are available at the prediction cutoff.

The functions are deliberately pure and do not fetch odds. They de-vig a complete
set of mutually exclusive decimal prices and fail closed on malformed inputs.
They must be evaluated with the odds' own publication/availability timestamp
strictly before the prediction cutoff.
"""
from __future__ import annotations

import math
from collections.abc import Iterable


def devig_probabilities(odds: Iterable[float]) -> list[float]:
    """Return normalized implied probabilities from a complete odds vector.

    No favorite-longshot correction or learned adjustment is applied here.
    Those are separate research hypotheses and must earn OOS evidence.
    """
    values = [float(x) for x in odds]
    if not values:
        raise ValueError("odds must not be empty")
    if any((not math.isfinite(x) or x <= 1.0) for x in values):
        raise ValueError("decimal odds must be finite and > 1.0")
    inv = [1.0 / x for x in values]
    total = sum(inv)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("invalid implied-probability mass")
    return [p / total for p in inv]
