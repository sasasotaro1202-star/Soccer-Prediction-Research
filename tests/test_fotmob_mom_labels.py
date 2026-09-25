import json

import pandas as pd
import pytest

from src.data.fotmob_mom_labels import (
    _extract_matches,
    _extract_potm,
    collect_fotmob_mom_labels,
    reconcile_fotmob_player_ids,
)


def test_extract_matches_supports_nested_league_fixture_shape():
    payload = {
        "matches": {
            "allMatches": [
                {
                    "id": 123,
                    "home": {"name": "Arsenal"},
                    "away": {"name": "Chelsea"},
                    "status": {"utcTime": "2025-01-15T19:00:00.000Z"},
                },
                {
                    "id": 123,
                    "home": {"name": "Arsenal"},
                    "away": {"name": "Chelsea"},
                    "status": {"utcTime": "2025-01-15T19:00:00.000Z"},
                },
            ]
        }
    }
    rows = _extract_matches(payload)
    assert len(rows) == 1
    assert rows[0]["match_id"] == "123"
    assert rows[0]["home_team"] == "Arsenal"


def test_extract_potm_uses_fotmob_matchfacts_shape():
    payload = {
        "content": {
            "matchFacts": {
                "playerOfTheMatch": {
                    "id": 456,
                    "name": {"fullName": "Bukayo Saka"},
                    "team": {"name": "Arsenal"},
                }
            }
        }
    }
    assert _extract_potm(payload) == ("456", "Bukayo Saka", "Arsenal")


def test_reconcile_fotmob_player_ids_requires_exact_single_candidate():
    labels = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "label_status": "LABEL_FOUND",
                "player_name": "Bukayo Saka",
                "player_provider_id": "456",
            },
            {
                "match_id": "m2",
                "label_status": "LABEL_FOUND",
                "player_name": "Unknown",
                "player_provider_id": "999",
            },
        ]
    )
    features = pd.DataFrame(
        [
            {"match_id": "m1", "player_id": "101", "player_name": "Bukayo Saka"},
            {"match_id": "m1", "player_id": "102", "player_name": "Other"},
        ]
    )
    out, report = reconcile_fotmob_player_ids(labels, features)
    assert out.loc[0, "player_id"] == "101"
    assert out.loc[1, "label_status"] == "PLAYER_NOT_RECONCILED"
    assert report == {"resolved": 1, "not_reconciled": 1, "ambiguous": 0}


def test_reconcile_fotmob_player_ids_supports_initial_and_compound_surname_aliases():
    labels = pd.DataFrame(
        [
            {"match_id": "m1", "label_status": "LABEL_FOUND", "player_name": "Danny Welbeck"},
            {"match_id": "m2", "label_status": "LABEL_FOUND", "player_name": "Emile Smith Rowe"},
        ]
    )
    features = pd.DataFrame(
        [
            {"match_id": "m1", "player_id": "101", "player_name": "D. Welbeck"},
            {"match_id": "m2", "player_id": "202", "player_name": "E. Smith Rowe"},
        ]
    )
    out, report = reconcile_fotmob_player_ids(labels, features)
    assert out["player_id"].tolist() == ["101", "202"]
    assert report == {"resolved": 2, "not_reconciled": 0, "ambiguous": 0}


def test_reconcile_fotmob_player_ids_fails_closed_on_ambiguous_name():
    labels = pd.DataFrame(
        [{"match_id": "m1", "label_status": "LABEL_FOUND", "player_name": "Alex"}]
    )
    features = pd.DataFrame(
        [
            {"match_id": "m1", "player_id": "101", "player_name": "Alex"},
            {"match_id": "m1", "player_id": "202", "player_name": "Alex"},
        ]
    )
    out, report = reconcile_fotmob_player_ids(labels, features)
    assert out.loc[0, "label_status"] == "AMBIGUOUS_PLAYER_NAME"
    assert report == {"resolved": 0, "not_reconciled": 0, "ambiguous": 1}


class _FakeFetcher:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.i = 0

    def get(self, source, url, **kwargs):
        body = json.dumps(self.payloads[self.i]).encode()
        self.i += 1

        class Meta:
            retrieved_at = "2026-01-01T00:00:00Z"
            cache_hit = True

        class Resp:
            def __init__(self, body):
                self.body = body
                self.metadata = Meta()

        return Resp(body)


def test_collect_fotmob_mom_labels_matches_and_extracts_potm(monkeypatch):
    import src.data.fotmob_mom_labels as mod

    fetcher = _FakeFetcher(
        [
            {
                "matches": [
                    {
                        "id": 123,
                        "home": {"name": "Arsenal"},
                        "away": {"name": "Chelsea"},
                        "status": {"utcTime": "2025-01-15T19:00:00Z"},
                    }
                ]
            },
            {
                "content": {
                    "matchFacts": {
                        "playerOfTheMatch": {
                            "id": 456,
                            "name": {"fullName": "Bukayo Saka"},
                            "team": {"name": "Arsenal"},
                        }
                    }
                }
            },
        ]
    )
    monkeypatch.setattr(mod, "ExternalFetcher", lambda **kwargs: fetcher)

    fixtures = pd.DataFrame(
        [
            {
                "match_id": "dataset-1",
                "kickoff_utc": "2025-01-15T19:00:00Z",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
            }
        ]
    )
    out = collect_fotmob_mom_labels(fixtures, request_delay_seconds=0)
    assert len(out) == 1
    assert out.loc[0, "label_status"] == "LABEL_FOUND"
    assert out.loc[0, "event_id"] == "123"
    assert out.loc[0, "player_provider_id"] == "456"
