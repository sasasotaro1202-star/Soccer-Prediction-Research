import scripts.run_mom_research as runner


def test_scope_resolver_uses_event_fallbacks_when_registries_fail(monkeypatch):
    def registry_tournaments(**kwargs):
        raise RuntimeError("registry unavailable")

    def event_tournament(dates_utc, **kwargs):
        assert dates_utc == ["2024-08-01", "2025-05-31"]
        return 17, {
            "method": "SCHEDULED_EVENTS_UNIQUE_TOURNAMENT",
            "independent_date_count": 2,
        }

    def registry_seasons(*args, **kwargs):
        raise RuntimeError("season registry unavailable")

    def event_season(dates_utc, **kwargs):
        assert dates_utc == ["2024-08-01", "2025-05-31"]
        assert kwargs["tournament_id"] == 17
        assert kwargs["season_start_year"] == 2024
        return 61627, {
            "method": "SCHEDULED_EVENTS_SEASON_FALLBACK",
            "independent_date_count": 2,
        }

    monkeypatch.setattr(runner, "fetch_unique_football_tournaments", registry_tournaments)
    monkeypatch.setattr(runner, "discover_unique_tournament_from_scheduled_events", event_tournament)
    monkeypatch.setattr(runner, "fetch_unique_tournament_seasons", registry_seasons)
    monkeypatch.setattr(runner, "discover_unique_season_from_scheduled_events", event_season)

    tournament_id, season_id, meta = runner._resolve_sofascore_scope(
        "EPL",
        2024,
        tournament_id=None,
        season_id=None,
        fetcher=object(),
        discovery_dates_utc=["2024-08-01", "2025-05-31"],
    )

    assert tournament_id == 17
    assert season_id == 61627
    assert meta["tournament_discovery"] == "DYNAMIC_EVENT_UNIQUE_TOURNAMENT_FALLBACK"
    assert meta["event_discovery"]["independent_date_count"] == 2
    assert meta["season_discovery"]["independent_date_count"] == 2
