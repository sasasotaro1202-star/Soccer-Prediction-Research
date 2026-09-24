from src.data.competition_catalog import ACTIVE_SCOPE, active_competitions
from src.data.competition_sources import TARGET_COMPETITIONS
from src.data.competition_sources import PLANS, TARGET_COMPETITIONS

EXPECTED = set(TARGET_COMPETITIONS)


def test_active_scope_matches_the_declared_target_scope():
    assert set(ACTIVE_SCOPE) == EXPECTED
    assert set(TARGET_COMPETITIONS) == EXPECTED
    assert {x.code for x in active_competitions()} == EXPECTED
    assert len(active_competitions()) == len(EXPECTED)


def test_every_active_competition_has_source_candidates():
    assert set(PLANS) >= EXPECTED
    for code in EXPECTED:
        assert PLANS[code]
