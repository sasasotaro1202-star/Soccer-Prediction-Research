import pytest

from src.models.mom_selector import select_mom_candidates


def test_selects_exactly_four_and_preserves_raw_probability():
    result = select_mom_candidates(
        ["p1", "p2", "p3", "p4", "p5"],
        [0.10, 0.30, 0.05, 0.40, 0.15],
    )
    assert [x.player_id for x in result] == ["p4", "p2", "p5", "p1"]
    assert [x.rank for x in result] == [1, 2, 3, 4]
    assert [x.probability for x in result] == [0.40, 0.30, 0.15, 0.10]
    assert sum(x.probability for x in result) == pytest.approx(0.95)


def test_production_output_is_fixed_to_four():
    with pytest.raises(ValueError):
        select_mom_candidates(["p1", "p2", "p3", "p4"], [1, 1, 1, 1], top_k=3)


def test_requires_four_eligible_players():
    with pytest.raises(ValueError):
        select_mom_candidates(["p1", "p2", "p3"], [1, 1, 1])


def test_rejects_duplicate_player_ids_within_fixture():
    with pytest.raises(ValueError):
        select_mom_candidates(["p1", "p1", "p2", "p3"], [0.4, 0.3, 0.2, 0.1])
