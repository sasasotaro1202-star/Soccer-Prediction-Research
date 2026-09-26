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


def test_timestamped_and_long_running_forecast_sources_are_catalogued():
    assert get_source("soccerportalx").source_class == "forecast"
    assert get_source("foresportia").source_class == "forecast"
    assert get_source("tofiko").source_class == "forecast"
    assert get_source("prosoccer").source_class == "forecast"
    assert get_source("the_football_simulator").source_class == "forecast"
    assert get_source("soccerportalx").production_status == "RESEARCH_ONLY"
    assert get_source("foresportia").production_status == "RESEARCH_ONLY"


def test_broad_external_prediction_and_stats_sources_are_research_scoped():
    assert get_source("betbrain").production_status == "RESEARCH_ONLY"
    assert get_source("soccervista").production_status == "RESEARCH_ONLY"
    assert get_source("footystats").production_status == "RESEARCH_CANDIDATE"
    assert get_source("soccerway").production_status == "RESEARCH_CANDIDATE"
    assert get_source("global_sports_archive").production_status == "RESEARCH_CANDIDATE"
    assert get_source("worldfootball_net").production_status == "RESEARCH_CANDIDATE"


def test_football_data_is_not_automated_under_current_source_terms():
    source = get_source("football_data")
    assert source.production_status == "RESEARCH_MANUAL_ONLY"
    assert source.automation_policy == "DO_NOT_AUTOMATE_WITHOUT_PERMISSION"


def test_reep_is_catalogued_as_a_free_cross_provider_identity_layer():
    source = get_source("reep")
    assert source.source_class == "entity_data"
    assert source.automation_policy == "FREE_DOWNLOAD_CC0"
    assert "Opta" in " ".join(source.strengths)


def test_open_event_tracking_and_womens_sources_are_catalogued():
    assert get_source("wyscout_pappalardo").source_class == "performance_data"
    assert get_source("impect_open").source_class == "performance_data"
    assert get_source("skillcorner_open").source_class == "tracking_data"
    assert get_source("metrica_sample").source_class == "tracking_data"
    assert get_source("soccer_mon").source_class == "tracking_data"
    assert get_source("ewf_database").source_class == "archive"
    assert get_source("brazilian_football_data").source_class == "archive"
    assert get_source("american_soccer_analysis").source_class == "performance_data"


def test_identity_and_video_benchmark_sources_are_not_match_time_features_by_default():
    assert get_source("fpl_id_map").source_class == "fantasy_data"
    assert get_source("wikidata_football").source_class == "entity_data"
    assert get_source("soccer_net").source_class == "tooling"
    assert get_source("fpl_id_map").pit_default.startswith("IDENTITY_MAP_ONLY")
    assert get_source("wikidata_football").pit_default.startswith("REFERENCE_GRAPH")


def test_large_archives_and_public_domain_world_data_are_catalogued():
    assert get_source("schochastics_football_data").source_class == "archive"
    assert get_source("international_results_cc0").source_class == "archive"
    assert get_source("fjelstul_english_football").source_class == "archive"
    assert get_source("footballcsv_world").source_class == "fixture_data"


def test_event_modelling_and_discovery_tooling_are_catalogued():
    assert get_source("kloppy").source_class == "tooling"
    assert get_source("socceraction").source_class == "tooling"
    assert get_source("pysport_index").source_class == "tooling"
    assert get_source("kloppy").production_status == "RESEARCH_CANDIDATE"
    assert get_source("socceraction").production_status == "RESEARCH_CANDIDATE"
    assert get_source("pysport_index").production_status == "DISCOVERY_SOURCE"


def test_live_data_and_news_channels_are_catalogued():
    assert get_source("football_data_org").source_class == "live_data"
    assert get_source("espn_soccer_scoreboard").source_class == "live_data"
    assert get_source("gdelt_doc").source_class == "news"
    assert get_source("gdelt_gkg").source_class == "news"
    assert get_source("bbc_football_rss").source_class == "news"
    assert get_source("gdelt_doc").production_status == "RESEARCH_CANDIDATE_FREE"


def test_official_asian_and_japanese_womens_sources_are_catalogued():
    assert get_source("kleague_official").source_class == "live_data"
    assert get_source("weleague_official").source_class == "live_data"
    assert get_source("kleague_official").automation_policy == "VERIFY_OFFICIAL_ACCESS_TERMS"
    assert get_source("weleague_official").automation_policy == "VERIFY_OFFICIAL_ACCESS_TERMS"


def test_regional_and_high_coverage_sources_are_catalogued():
    assert get_source("playmakerstats").source_class == "fixture_data"
    assert get_source("bdfutbol").source_class == "fixture_data"
    assert get_source("jleague_official").source_class == "live_data"
    assert get_source("kooora").source_class == "live_data"
    assert get_source("jleague_official").automation_policy == "VERIFY_OFFICIAL_ACCESS_TERMS"


def test_broad_live_providers_and_multi_source_tooling_are_catalogued():
    assert get_source("sofascore_public").source_class == "live_data"
    assert get_source("fotmob_public").source_class == "live_data"
    assert get_source("soccerdata_python").source_class == "tooling"
    assert get_source("worldfootballr").source_class == "tooling"
    assert get_source("kaggle_european_soccer_db").source_class == "archive"
    assert get_source("sofascore_public").production_status == "RESEARCH_CANDIDATE"


def test_odds_and_referee_channels_are_catalogued():
    assert get_source("odds_api_io").source_class == "odds_market"
    assert get_source("pulsescore_odds").source_class == "odds_market"
    assert get_source("statsbet_referees").source_class == "performance_data"
    assert get_source("refsradar").source_class == "performance_data"
    assert get_source("scorelineai_referees").source_class == "performance_data"
    assert get_source("odds_api_io").production_status == "RESEARCH_CANDIDATE_FREE_TIER"


def test_free_api_fixture_and_lineup_candidates_are_catalogued():
    assert get_source("openfootapi").source_class == "live_data"
    assert get_source("bsd_free_football_api").source_class == "live_data"
    assert get_source("openfootapi").production_status == "RESEARCH_CANDIDATE"
    assert get_source("bsd_free_football_api").production_status == "RESEARCH_CANDIDATE_FREE"


def test_venue_enrichment_sources_are_catalogued():
    assert get_source("openfootball_clubs_stadiums").source_class == "venue_data"
    assert get_source("world_soccer_stadiums").source_class == "venue_data"
    assert get_source("world_soccer_stadiums").production_status == "RESEARCH_CANDIDATE"


def test_global_data_lake_and_open_womens_event_sources_are_catalogued():
    assert get_source("global_football_data_lake").source_class == "archive"
    assert get_source("global_football_data_lake").production_status == "RESEARCH_CANDIDATE"
    assert get_source("dynasty_scouting_league_2024").source_class == "performance_data"
    assert get_source("wosostats").source_class == "performance_data"


def test_national_team_strength_sources_are_catalogued_separately():
    assert get_source("world_football_elo").source_class == "rating"
    assert get_source("fifa_world_ranking").source_class == "rating"


def test_free_fixture_fallbacks_are_explicitly_research_scoped():
    assert get_source("football_data").production_status == "RESEARCH_MANUAL_ONLY"
    assert get_source("openligadb").production_status == "RESEARCH_CANDIDATE"
    assert get_source("api_football_free").production_status == "RESEARCH_CANDIDATE_FREE_TIER_ONLY"
    assert get_source("thesportsdb_v1").production_status == "RESEARCH_CANDIDATE"


def test_rsssf_is_archive_only():
    source = get_source("rsssf")
    assert source.source_class == "fixture_data"
    assert source.pit_default.startswith("ARCHIVAL_RESULTS_ONLY")
