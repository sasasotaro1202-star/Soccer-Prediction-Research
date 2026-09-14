import pytest

from src.models.score_selector import select_score_candidates


def test_selects_exactly_three_scores():
    result = select_score_candidates(
        [1, 1, 2, 0], [0, 1, 1, 0], [0.20, 0.35, 0.30, 0.15]
    )
    assert [(x.home_goals, x.away_goals) for x in result] == [(1, 1), (2, 1), (1, 0)]
    assert [x.rank for x in result] == [1, 2, 3]
    assert sum(x.probability for x in result) == pytest.approx(1.0)


def test_score_output_is_fixed_to_three():
    with pytest.raises(ValueError):
        select_score_candidates([1, 0, 2], [0, 0, 1], [0.4, 0.3, 0.3], top_k=2)
