from src.data.competition_catalog import COMPETITION_CATALOG
from src.data.competition_sources import catalog_source_plans


def test_catalog_contains_multiple_non_league_formats():
    kinds = {spec.competition_type for spec in COMPETITION_CATALOG}
    assert {"league", "cup", "league_cup", "super_cup", "international", "friendly", "multi_sport", "youth"} <= kinds


def test_catalog_source_plan_covers_every_catalogued_competition():
    plans = {p.competition: p for p in catalog_source_plans()}
    assert len(plans) == len(COMPETITION_CATALOG)
    for spec in COMPETITION_CATALOG:
        assert spec.code in plans
        assert plans[spec.code].canonical_candidates
