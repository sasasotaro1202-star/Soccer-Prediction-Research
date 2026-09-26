from src.research.external_source_catalog import (
    SOURCE_CATALOG,
    get_source,
    sources_by_class,
)


def test_catalog_has_broad_forecast_and_enrichment_classes():
    classes = {source.source_class for source in SOURCE_CATALOG}
    assert {"forecast", "rating", "performance_data", "player_status", "weather"} <= classes


def test_broad_forecast_sources_are_research_only():
    keys = {source.key for source in sources_by_class("forecast")}
    assert {"opta_analyst", "forebet", "predictz", "oddspedia_smartbet", "fivethirtyeight_spi"} <= keys
    assert all(get_source(key).production_status == "RESEARCH_ONLY" for key in keys)


def test_pit_unknown_external_forecasts_are_fail_closed_by_default_metadata():
    assert get_source("forebet").pit_default == "UNKNOWN_FOR_HISTORICAL_REPLAY"
    assert get_source("predictz").pit_default == "UNKNOWN_FOR_HISTORICAL_REPLAY"
    assert get_source("opta_analyst").pit_default == "UNKNOWN_UNLESS_DOCUMENTED"


def test_open_meteo_is_marked_for_timestamped_historical_forecast_reconstruction():
    source = get_source("open_meteo")
    assert source.source_class == "weather"
    assert "FORECAST_RUN_TIMESTAMP_AVAILABLE" in source.pit_default
