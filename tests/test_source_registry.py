from src.data.source_registry import SOCCER_SOURCES, get_source, pit_sources, source_names


def test_registry_has_multiple_independent_source_families():
    assert len(SOCCER_SOURCES) >= 8
    assert len({s.kind for s in SOCCER_SOURCES}) >= 4


def test_registry_has_two_pit_capable_archive_sources():
    names = {s.name for s in pit_sources()}
    assert "Internet Archive / Wayback" in names
    assert "Arquivo.pt" in names


def test_source_metadata_is_explicit():
    for source in SOCCER_SOURCES:
        assert source.name
        assert source.kind
        assert source.role
        assert source.fields
        assert isinstance(source.historical, bool)
        assert isinstance(source.pit_capable, bool)
        assert isinstance(source.live_capable, bool)
        assert isinstance(source.auth_required, bool)


def test_lookup_and_names_are_stable():
    assert get_source("Sofascore").kind == "public_api"
    assert "API-Football" in source_names()
    assert "Understat" in source_names()


def test_versioned_serie_a_source_is_explicitly_pit_capable():
    source = get_source("footballcsv/cache.footballdata (versioned Italy 2019-20/2020-21 snapshots)")
    assert source.pit_capable is True
    assert source.auth_required is False
    assert "SA" in source.primary_for


def test_versioned_la_liga_source_is_explicitly_pit_capable():
    source = get_source("footballcsv/espana (versioned 2019-20/2020-21 snapshots)")
    assert source.pit_capable is True
    assert source.auth_required is False
    assert "LL" in source.primary_for


def test_versioned_j1_2020_source_is_explicitly_pit_capable():
    source = get_source(
        "ewalldo/Japan-J1-League-Data-and-Data-Analysis (versioned J1 2020 snapshot)"
    )
    assert source.pit_capable is True
    assert source.auth_required is False
