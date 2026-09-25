import json

import pandas as pd
import pytest

from src.data.sofascore_mom_labels import (
    _event_candidates,
    _extract_player_of_match,
    _norm_team,
    fetch_unique_football_tournaments,
    resolve_unique_tournament_id,
    discover_unique_tournament_from_scheduled_events,
    discover_unique_season_from_scheduled_events,
    collect_sofascore_mom_labels,
    fetch_tournament_season_events_by_rounds,
    label_data_contract_report,
)


def test_team_normalization_is_conservative():
    assert _norm_team("Arsenal FC") == "arsenalfc"
    assert _norm_team("Paris Saint-Germain") == "parissaintgermain"
    assert _norm_team("Málaga") == "malaga"
    assert _norm_team("鹿島アントラーズ") == "鹿島アントラーズ"


def test_extract_player_of_match_accepts_nested_player_shape():
    player_id, name = _extract_player_of_match({
        "playerOfTheMatch": {"player": {"id": 123, "name": "Test Player"}}
    })
    assert player_id == "123"
    assert name == "Test Player"


def test_extract_player_of_match_rejects_missing_label():
    assert _extract_player_of_match({"bestHome": {"id": 123}}) == ("", "")


def test_event_match_requires_exact_team_names_and_near_kickoff():
    fixture = pd.Series({
        "home_team": "Home FC",
        "away_team": "Away FC",
        "kickoff_utc": "2026-09-25T12:00:00Z",
    })
    events = [
        {
            "id": 1,
            "startTimestamp": int(pd.Timestamp("2026-09-25T12:05:00Z").timestamp()),
            "homeTeam": {"name": "Home FC"},
            "awayTeam": {"name": "Away FC"},
        },
        {
            "id": 2,
            "startTimestamp": int(pd.Timestamp("2026-09-25T12:05:00Z").timestamp()),
            "homeTeam": {"name": "Home FC"},
            "awayTeam": {"name": "Other FC"},
        },
    ]
    matches = _event_candidates(events, fixture, 1.0)
    assert len(matches) == 1
    assert matches[0]["event"]["id"] == 1


def test_event_match_rejects_far_kickoff():
    fixture = pd.Series({
        "home_team": "Home FC",
        "away_team": "Away FC",
        "kickoff_utc": "2026-09-25T12:00:00Z",
    })
    events = [{
        "id": 1,
        "startTimestamp": int(pd.Timestamp("2026-09-25T20:00:00Z").timestamp()),
        "homeTeam": {"name": "Home FC"},
        "awayTeam": {"name": "Away FC"},
    }]
    assert _event_candidates(events, fixture, 1.0) == []


def test_collect_labels_keeps_missing_label_as_missing(monkeypatch):
    import src.data.sofascore_mom_labels as m

    class FakeResponse:
        def __init__(self, payload):
            self.body = json.dumps(payload).encode()
            self.metadata = type("Meta", (), {"retrieved_at": "2026-09-25T13:00:00Z", "cache_hit": False})()

    class FakeFetcher:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, source, url, **kwargs):
            if "scheduled-events" in url:
                return FakeResponse({
                    "events": [{
                        "id": 12345,
                        "startTimestamp": int(pd.Timestamp("2026-09-25T12:00:00Z").timestamp()),
                        "homeTeam": {"name": "Home FC"},
                        "awayTeam": {"name": "Away FC"},
                    }]
                })
            return FakeResponse({"bestHome": {"player": {"id": 999, "name": "Other"}}})

    monkeypatch.setattr(m, "ExternalFetcher", FakeFetcher)
    fixtures = pd.DataFrame([{
        "match_id": "m1",
        "kickoff_utc": "2026-09-25T12:00:00Z",
        "home_team": "Home FC",
        "away_team": "Away FC",
    }])
    labels = collect_sofascore_mom_labels(fixtures)
    assert labels.loc[0, "label_status"] == "LABEL_MISSING"
    assert labels.loc[0, "player_id"] == ""


def test_collect_labels_marks_ambiguous_event_match(monkeypatch):
    import src.data.sofascore_mom_labels as m

    class FakeResponse:
        def __init__(self, payload):
            self.body = json.dumps(payload).encode()
            self.metadata = type("Meta", (), {"retrieved_at": "2026-09-25T13:00:00Z"})()

    class FakeFetcher:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, source, url, **kwargs):
            return FakeResponse({
                "events": [
                    {
                        "id": 1,
                        "startTimestamp": int(pd.Timestamp("2026-09-25T12:00:00Z").timestamp()),
                        "homeTeam": {"name": "Home FC"},
                        "awayTeam": {"name": "Away FC"},
                    },
                    {
                        "id": 2,
                        "startTimestamp": int(pd.Timestamp("2026-09-25T12:00:00Z").timestamp()),
                        "homeTeam": {"name": "Home FC"},
                        "awayTeam": {"name": "Away FC"},
                    },
                ]
            })

    monkeypatch.setattr(m, "ExternalFetcher", FakeFetcher)
    fixtures = pd.DataFrame([{
        "match_id": "m1",
        "kickoff_utc": "2026-09-25T12:00:00Z",
        "home_team": "Home FC",
        "away_team": "Away FC",
    }])
    labels = collect_sofascore_mom_labels(fixtures)
    assert labels.loc[0, "label_status"] == "AMBIGUOUS_EVENT_MATCH"


def test_label_contract_does_not_treat_missing_as_negative():
    report = label_data_contract_report(pd.DataFrame([{
        "match_id": "m1",
        "label_status": "LABEL_MISSING",
    }]))
    assert report["label_found"] == 0
    assert report["label_missing"] == 1
    assert report["fail_closed"] is True


def test_mom_label_uses_best_players_summary_endpoint(monkeypatch):
    import src.data.sofascore_mom_labels as m

    class FakeResponse:
        body = json.dumps({"playerOfTheMatch": {"player": {"id": 123, "name": "Winner"}}}).encode()
        metadata = type("Meta", (), {"retrieved_at": "2026-09-25T13:00:00Z"})()

    seen = []
    class FakeFetcher:
        def get(self, source, url, **kwargs):
            seen.append(url)
            return FakeResponse()

    result = m.fetch_mom_label(987, fetcher=FakeFetcher())
    assert result["label_found"] is True
    assert result["player_id"] == "123"
    assert seen == ["https://api.sofascore.com/api/v1/event/987/best-players/summary"]


def test_request_delay_rejects_non_finite_value():
    import src.data.sofascore_mom_labels as m
    class FakeFetcher:
        def __init__(self, *args, **kwargs):
            pass
    monkey = None
    fixtures = pd.DataFrame([{
        "match_id": "m1",
        "kickoff_utc": "2026-09-25T12:00:00Z",
        "home_team": "Home FC",
        "away_team": "Away FC",
    }])
    # Invalid delay is rejected before any network call.
    with pytest.raises(ValueError, match="finite and non-negative"):
        m.collect_sofascore_mom_labels_tournament_season(
            fixtures,
            tournament_id=17,
            season_id=100,
            request_delay_seconds=float("inf"),
        )


def test_select_season_id_requires_unique_match():
    from src.data.sofascore_mom_labels import select_season_id
    assert select_season_id([{"id": 2024, "name": "2024/25"}], 2024) == 2024
    with pytest.raises(RuntimeError, match="ambiguous"):
        select_season_id([{"id": 1, "name": "2024"}, {"id": 2, "name": "2024/25"}], 2024)


def test_fetch_season_events_stops_after_empty_page(monkeypatch):
    import src.data.sofascore_mom_labels as m

    class Response:
        def __init__(self, payload):
            self.body = json.dumps(payload).encode()
            self.metadata = type("Meta", (), {"retrieved_at": "2026-09-25T13:00:00Z", "cache_hit": False})()

    class FakeFetcher:
        def __init__(self):
            self.calls = []
        def get(self, source, url, **kwargs):
            self.calls.append(url)
            page = int(url.rsplit("/", 1)[-1])
            payload = {"events": []} if page >= 2 else {"events": [{"id": page + 1}]}
            return Response(payload)

    fetcher = FakeFetcher()
    events, retrieved = m.fetch_tournament_season_events(
        17, 100, fetcher=fetcher, max_pages=10
    )
    assert [x["id"] for x in events] == [1, 2]
    assert len(fetcher.calls) == 3
    assert retrieved == "2026-09-25T13:00:00Z"


def test_round_event_fallback_collects_until_consecutive_empty_rounds(monkeypatch):
    import src.data.sofascore_mom_labels as m

    class Response:
        def __init__(self, payload):
            self.body = json.dumps(payload).encode()
            self.metadata = type(
                "Meta", (), {"retrieved_at": "2026-09-25T13:00:00Z", "cache_hit": False}
            )()

    class FakeFetcher:
        def __init__(self):
            self.calls = []

        def get(self, source, url, **kwargs):
            self.calls.append(url)
            round_number = int(url.rsplit("/", 1)[-1])
            payload = (
                {"events": [{"id": 101}, {"id": 102}]}
                if round_number == 1
                else {"events": []}
            )
            return Response(payload)

    fetcher = FakeFetcher()
    events, retrieved = fetch_tournament_season_events_by_rounds(
        17, 61627, fetcher=fetcher, max_rounds=10, request_delay_seconds=0
    )
    assert [x["id"] for x in events] == [101, 102]
    assert len(fetcher.calls) == 4
    assert all("/events/round/" in url for url in fetcher.calls)
    assert retrieved == "2026-09-25T13:00:00Z"


def test_round_event_fallback_fails_closed_on_missing_payload():
    import src.data.sofascore_mom_labels as m

    class Response:
        body = json.dumps({"unexpected": []}).encode()
        metadata = type("Meta", (), {"retrieved_at": "2026-09-25T13:00:00Z"})()

    class FakeFetcher:
        def get(self, source, url, **kwargs):
            return Response()

    with pytest.raises(RuntimeError, match="round 1 acquisition failed"):
        m.fetch_tournament_season_events_by_rounds(
            17, 61627, fetcher=FakeFetcher(), max_rounds=2, request_delay_seconds=0
        )


def test_tournament_resolver_is_exact_and_category_aware():
    tournaments = [
        {"id": 1, "name": "Premier League", "category": {"name": "England"}},
        {"id": 2, "name": "Premier League", "category": {"name": "Other"}},
    ]
    assert resolve_unique_tournament_id(
        tournaments,
        names=["Premier League"],
        category_names=["England"],
    ) == 1
    with pytest.raises(RuntimeError, match="ambiguous or missing"):
        resolve_unique_tournament_id(tournaments, names=["Premier League"])


def test_tournament_registry_extraction_fails_closed_without_list(monkeypatch):
    import src.data.sofascore_mom_labels as m

    class Response:
        body = json.dumps({"unexpected": []}).encode()
        metadata = type("Meta", (), {"retrieved_at": "2026-09-25T13:00:00Z"})()

    class FakeFetcher:
        def get(self, source, url, **kwargs):
            return Response()

    with pytest.raises(RuntimeError, match="discovery failed"):
        fetch_unique_football_tournaments(fetcher=FakeFetcher())


def test_event_based_tournament_discovery_requires_two_dates_and_exact_name():
    import src.data.sofascore_mom_labels as m

    class Response:
        def __init__(self, payload):
            self.body = json.dumps(payload).encode()
            self.metadata = type("Meta", (), {"retrieved_at": "2026-09-25T13:00:00Z"})()

    class FakeFetcher:
        def __init__(self):
            self.calls = []

        def get(self, source, url, **kwargs):
            self.calls.append(url)
            date = url.rsplit("/", 1)[-1]
            if date in {"2024-08-01", "2025-05-31"}:
                payload = {"events": [{
                    "id": 100 + len(self.calls),
                    "uniqueTournament": {
                        "id": 17,
                        "name": "Premier League",
                        "slug": "premier-league",
                        "category": {"name": "England"},
                    },
                }]}
            else:
                payload = {"events": [{
                    "id": 200 + len(self.calls),
                    "uniqueTournament": {
                        "id": 999,
                        "name": "Premier League 2",
                        "slug": "premier-league-2",
                        "category": {"name": "England"},
                    },
                }]}
            return Response(payload)

    tid, meta = discover_unique_tournament_from_scheduled_events(
        ["2024-08-01", "2025-01-01", "2025-05-31"],
        names=["Premier League"],
        category_names=["England"],
        fetcher=FakeFetcher(),
        max_dates=3,
    )
    assert tid == 17
    assert meta["independent_date_count"] == 2


def test_event_based_tournament_discovery_fails_closed_when_multiple_dates_do_not_repeat():
    class Response:
        def __init__(self, tournament_id):
            self.body = json.dumps({"events": [{
                "uniqueTournament": {
                    "id": tournament_id,
                    "name": "Premier League",
                    "category": {"name": "England"},
                }
            }]}).encode()
            self.metadata = type("Meta", (), {"retrieved_at": "2026-09-25T13:00:00Z"})()

    class FakeFetcher:
        def get(self, source, url, **kwargs):
            date = url.rsplit("/", 1)[-1]
            tournament_id = 17 if date == "2024-08-01" else 18
            return Response(tournament_id)

    with pytest.raises(RuntimeError, match="ambiguous or insufficiently repeated"):
        discover_unique_tournament_from_scheduled_events(
            ["2024-08-01", "2025-01-01"],
            names=["Premier League"],
            fetcher=FakeFetcher(),
            max_dates=2,
        )



def test_event_based_season_discovery_requires_two_dates():
    class Response:
        def __init__(self, payload):
            self.body = json.dumps(payload).encode()
            self.metadata = type("Meta", (), {"retrieved_at": "2026-09-25T13:00:00Z"})()

    class FakeFetcher:
        def get(self, source, url, **kwargs):
            date = url.rsplit("/", 1)[-1]
            season = (
                {"id": 61627, "name": "2024/2025", "year": 2024}
                if date in {"2024-08-01", "2025-05-31"}
                else {"id": 99999, "name": "2023/2024", "year": 2023}
            )
            return Response({"events": [{
                "id": 1,
                "uniqueTournament": {"id": 17, "name": "Premier League"},
                "season": season,
            }]})

    tid, meta = discover_unique_season_from_scheduled_events(
        ["2024-08-01", "2025-01-01", "2025-05-31"],
        tournament_id=17,
        season_start_year=2024,
        fetcher=FakeFetcher(),
        max_dates=3,
    )
    assert tid == 61627
    assert meta["independent_date_count"] == 2


def test_event_based_tournament_discovery_accepts_exact_singleton_competition():
    class Response:
        body = json.dumps({"events": [{
            "uniqueTournament": {
                "id": 700,
                "name": "UEFA Super Cup",
                "category": {"name": "Europe"},
            }
        }]}).encode()
        metadata = type("Meta", (), {"retrieved_at": "2026-09-25T13:00:00Z"})()

    class FakeFetcher:
        def get(self, source, url, **kwargs):
            return Response()

    tid, meta = discover_unique_tournament_from_scheduled_events(
        ["2024-08-14"],
        names=["UEFA Super Cup"],
        fetcher=FakeFetcher(),
        max_dates=2,
    )
    assert tid == 700
    assert meta["independent_date_count"] == 1
