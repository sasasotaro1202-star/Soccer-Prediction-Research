from __future__ import annotations

from src.data.competition_catalog import COMPETITION_CATALOG, production_candidates, research_targets
from src.data.competition_sources import AUXILIARY_COMPETITIONS, AUXILIARY_PLANS, PLANS, TARGET_COMPETITIONS


def test_catalog_codes_are_unique_and_research_scope_is_explicit():
    specs = list(COMPETITION_CATALOG)
    codes = [spec.code for spec in specs]
    assert len(codes) == len(set(codes))
    assert all(spec.code and spec.name and spec.region and spec.competition_type for spec in specs)
    assert all(spec.code in {s.code for s in research_targets()} for spec in specs if spec.research_enabled)


def test_unverified_broad_catalog_never_auto_promotes_to_production():
    for spec in COMPETITION_CATALOG:
        if spec.pit_status != "VERIFIED" or spec.data_quality_status != "VERIFIED":
            assert not spec.production_eligible
    # Production adoption is an explicit downstream gate, never an implication
    # of merely registering a competition as a research target.
    assert all(spec.production_eligible for spec in production_candidates())


def test_every_strict_or_auxiliary_competition_has_an_explicit_source_plan():
    planned = set(PLANS) | set(AUXILIARY_PLANS)
    assert set(TARGET_COMPETITIONS) <= planned
    assert set(AUXILIARY_COMPETITIONS) <= planned


def test_broad_catalog_targets_not_yet_backed_by_sources_remain_unverified():
    planned = set(PLANS) | set(AUXILIARY_PLANS)
    for spec in COMPETITION_CATALOG:
        if spec.code not in planned:
            assert spec.pit_status == "UNVERIFIED"
            assert spec.data_quality_status == "UNVERIFIED"
            assert not spec.production_eligible
