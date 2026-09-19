from __future__ import annotations

import pytest

from src.evaluation.market import devig_probabilities


def test_devig_normalizes_implied_probabilities_without_changing_order() -> None:
    probs = devig_probabilities([2.0, 3.0, 4.0])
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] > probs[1] > probs[2]


@pytest.mark.parametrize("odds", [[], [1.0, 2.0], [2.0, float("nan")], [2.0, float("inf")]])
def test_devig_fails_closed_on_invalid_odds(odds) -> None:
    with pytest.raises(ValueError):
        devig_probabilities(odds)
