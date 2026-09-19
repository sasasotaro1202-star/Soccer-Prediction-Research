from src.data.competition_catalog import ACTIVE_SCOPE, active_competitions
from src.data.competition_sources import PLANS, TARGET_COMPETITIONS

EXPECTED = {
    "EPL", "AG_M", "AG_W", "ERE", "LL", "SA", "BL1", "J1", "J2", "J3",
    "FL1", "UCL", "UEL", "U23_M", "U18_M",
}


def test_active_scope_is_exactly_the_requested_fifteen():
    assert set(ACTIVE_SCOPE) == EXPECTED
    assert set(TARGET_COMPETITIONS) == EXPECTED
    assert {x.code for x in active_competitions()} == EXPECTED
    assert len(active_competitions()) == 15


def test_every_active_competition_has_source_candidates():
    assert set(PLANS) >= EXPECTED
    for code in EXPECTED:
        assert PLANS[code]
