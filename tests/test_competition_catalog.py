from src.data.competition_catalog import COMPETITION_CATALOG, production_candidates, research_targets


def test_catalog_covers_requested_core_and_major_cup_scope():
    names = {spec.name for spec in COMPETITION_CATALOG}
    required = {
        "Premier League",
        "Championship",
        "Bundesliga",
        "Serie A",
        "La Liga",
        "Ligue 1",
        "Eredivisie",
        "UEFA Champions League",
        "UEFA Europa League",
        "J1 League",
        "J2 League",
        "J3 League",
        "DFB-Pokal",
        "International / Club Friendly",
        "EFL Cup / Carabao Cup",
        "FA Cup",
        "Copa del Rey",
        "Coppa Italia",
        "Coupe de France",
        "UEFA Conference League",
    }
    assert required <= names


def test_all_catalog_entries_are_research_targets_but_none_auto_promote():
    assert len(research_targets()) == len(COMPETITION_CATALOG)
    assert production_candidates() == []


def test_production_eligibility_is_explicit_and_fail_closed():
    for spec in COMPETITION_CATALOG:
        if spec.production_eligible:
            assert spec.pit_status == "VERIFIED"
            assert spec.data_quality_status == "PASSED"
