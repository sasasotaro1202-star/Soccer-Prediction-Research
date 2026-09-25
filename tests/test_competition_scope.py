from src.data.competition_sources import TARGET_COMPETITIONS, source_plans


def test_all_active_competitions_have_unique_source_plans():
    assert TARGET_COMPETITIONS
    assert len(TARGET_COMPETITIONS) == len(set(TARGET_COMPETITIONS))
    plans = source_plans()
    planned = {plan.competition for plan in plans}
    assert planned == set(TARGET_COMPETITIONS)
    assert all(plan.canonical_candidates for plan in plans)


def test_scope_contains_all_declared_active_competition_families():
    required = {
        "EPL", "ERE", "LL", "SA", "BL1", "FL1",
        "J1", "J2", "J3",
        "UCL", "UEL", "UECL", "UWCL",
        "UEFA_EURO_M", "UEFA_EURO_W",
        "UEFA_NATIONS_LEAGUE_M", "UEFA_NATIONS_LEAGUE_W",
        "INTL_M", "INTL_W",
        "U23_M", "U18_M",
    }
    assert required.issubset(set(TARGET_COMPETITIONS))
