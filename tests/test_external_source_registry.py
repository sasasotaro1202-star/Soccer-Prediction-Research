from src.data.external_source_registry import EXTERNAL_SOURCES, audit_sources, production_sources, source_spec


def test_external_source_registry_contains_expected_candidates():
    assert [spec.key for spec in audit_sources()] == [
        "clubelo",
        "statsbomb_open_data",
        "understat",
        "open_meteo",
        "sofascore",
        "api_football",
        "fbref",
        "sportmonks",
        "x_api",
        "instagram_graph_api",
        "facebook_graph_api",
    ]
    assert len(EXTERNAL_SOURCES) == 11


def test_external_sources_are_fail_closed_by_default():
    assert production_sources() == ()
    assert all(not spec.production_enabled for spec in EXTERNAL_SOURCES)
    assert all(spec.pit_status == "UNVERIFIED" for spec in EXTERNAL_SOURCES)


def test_source_lookup_is_stable():
    assert source_spec("clubelo").name == "ClubElo"
    assert source_spec("api_football").api_key_env == "API_FOOTBALL_KEY"
    assert source_spec("sportmonks").api_key_env == "SPORTMONKS_API_TOKEN"
