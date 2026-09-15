from src.models.score_selector import select_score_candidates


def test_selects_exactly_three_and_preserves_raw_probability():
    selected = select_score_candidates(
        [
            (0, 0, 0.20),
            (1, 0, 0.35),
            (0, 1, 0.30),
            (1, 1, 0.15),
        ]
    )

    assert len(selected) == 3
    assert [(x.home_goals, x.away_goals) for x in selected] == [(1, 0), (0, 1), (0, 0)]
    assert [x.probability for x in selected] == [0.35, 0.30, 0.20]
    assert sum(x.probability for x in selected) == 0.85


def test_rejects_invalid_top_k():
    try:
        select_score_candidates([(0, 0, 1.0), (1, 0, 0.0), (0, 1, 0.0)], top_k=2)
    except ValueError as exc:
        assert "exactly 3" in str(exc)
    else:
        raise AssertionError("expected ValueError")
