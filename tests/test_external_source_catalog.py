from src.research.external_source_catalog import (
    SOURCE_CATALOG,
    get_source,
    sources_by_class,
)


def test_catalog_covers_forecast_and_enrichment_classes():
    classes = {source.source_class for source in SOURCE_CATALOG}
    assert {
        "forecast",
        "rating",
        "performance_data",
        "player_status",
        "weather",
        "fixture_data",
        "entity_data",
    } <= classes


def test_broad_forecast_sources_are_research_only():
    keys = {source.key for source in sources_by_class("forecast")}
    assert {
        "opta_analyst",
        "forebet",
        "predictz",
        "oddspedia_smartbet",
        "fivethirtyeight_spi",
    } <= keys
    assert all(get_source(key).production_status == "RESEARCH_ONLY" for key in keys)


def test_pit_unknown_external_forecasts_are_fail_closed_by_default_metadata():
    assert get_source("forebet").pit_default == "UNKNOWN_FOR_HISTORICAL_REPLAY"
    assert get_source("predictz").pit_default == "UNKNOWN_FOR_HISTORICAL_REPLAY"
    assert get_source("opta_analyst").pit_default == "UNKNOWN_UNLESS_DOCUMENTED"


def test_open_meteo_is_marked_for_timestamped_historical_forecast_reconstruction():
    source = get_source("open_meteo")
    assert source.source_class == "weather"
    assert "FORECAST_RUN_TIMESTAMP_AVAILABLE" in source.pit_default


def test_openfootball_supports_scope_and_entity_expansion_without_claiming_match_time_features():
    assert get_source("openfootball_world").source_class == "fixture_data"
    assert get_source("openfootball_players").source_class == "entity_data"
    assert "FEATURE_AVAILABILITY" in get_source("openfootball_world").pit_default


def test_broad_external_prediction_and_stats_sources_are_research_scoped():
    assert get_source("betbrain").production_status == "RESEARCH_ONLY"
    assert get_source("soccervista").production_status == "RESEARCH_ONLY"
    assert get_source("footystats").production_status == "RESEARCH_CANDIDATE"
    assert get_source("soccerway").production_status == "RESEARCH_CANDIDATE"
    assert get_source("global_sports_archive").production_status == "RESEARCH_CANDIDATE"
    assert get_source("worldfootball_net").production_status == "RESEARCH_CANDIDATE"


def test_free_fixture_fallbacks_are_explicitly_research_scoped():
    assert get_source("football_data").production_status == "RESEARCH_CANDIDATE"
    assert get_source("openligadb").production_status == "RESEARCH_CANDIDATE"
    assert get_source("api_football_free").production_status == "RESEARCH_CANDIDATE_FREE_TIER_ONLY"
    assert get_source("thesportsdb_v1").production_status == "RESEARCH_CANDIDATE"


def test_rsssf_is_archive_only():
    source = get_source("rsssf")
    assert source.source_class == "fixture_data"
    assert source.pit_default.startswith("ARCHIVAL_RESULTS_ONLY")
