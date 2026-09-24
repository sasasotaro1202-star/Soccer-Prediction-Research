import json

import pandas as pd
import pytest

from src.data.sofascore_mom_labels import (
    _event_candidates,
    _extract_player_of_match,
    _norm_team,
    collect_sofascore_mom_labels,
    label_data_contract_report,
)


def test_team_normalization_is_conservative():
    assert _norm_team("Arsenal FC") == "arsenalfc"
    assert _norm_team("Paris Saint-Germain") == "parissaintgermain"
    assert _norm_team("Málaga") == "malaga"


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
            self.metadata = type("Meta", (), {"retrieved_at": "2026-09-25T13:00:00Z"})()

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
