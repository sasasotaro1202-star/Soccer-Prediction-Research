from src.research.opta_like_source_registry import (
    all_sources,
    data_sources_only,
    source_summary,
    usable_source_candidates,
)


def test_registry_has_broad_free_public_source_coverage():
    summary = source_summary()
    assert summary["total"] >= 15
    assert summary["data_sources"] >= 10
    assert summary["free_or_public"] >= 10
    assert summary["research_only_high_value"] >= 8


def test_priority_candidates_include_event_and_tracking_channels():
    tactics = usable_source_candidates(target_axis="tactics")
    keys = {source.key for source in tactics}
    assert {"statsbomb_open", "skillcorner_open", "laurie_on_tracking", "driblab_open", "idsse_dfl_open"} <= keys

    integration = usable_source_candidates(target_axis="data_integration")
    integration_keys = {source.key for source in integration}
    assert "kloppy" in integration_keys


def test_data_only_view_excludes_feature_engines():
    keys = {source.key for source in data_sources_only()}
    assert "socceraction" not in keys
    assert "kloppy" not in keys
    assert "statsbomb_open" in keys


def test_all_sources_are_explicit_about_pit_and_license():
    for source in all_sources():
        assert source.pit_status
        assert source.license_status
        assert source.source_url.startswith("https://")


def test_archived_and_free_plan_sources_are_explicitly_classified():
    sources = {source.key: source for source in all_sources()}
    assert sources["football_data_org"].access == "FREE_PLAN"
    assert sources["fivethirtyeight_spi"].pit_status == "HISTORICAL_ONLY"
    assert sources["clubelo_archive"].pit_status == "CURRENT_ACCESS_UNCONFIRMED"
