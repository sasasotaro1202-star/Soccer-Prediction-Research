import pandas as pd
import requests

from src.research.sofascore_official_potm import (
    attach_labels,
    fetch_official_potm,
    parse_player_of_match,
)


def test_parse_official_player_of_match():
    payload = {
        "playerOfTheMatch": {
            "value": "8.7",
            "label": "rating",
            "player": {"id": 1234, "name": "Example Player"},
        }
    }
    parsed = parse_player_of_match(payload)
    assert parsed == {
        "player_id": "1234",
        "player_name": "Example Player",
        "rating_value": 8.7,
        "rating_label": "rating",
    }


def test_parse_missing_player_of_match_returns_none():
    assert parse_player_of_match({"bestHomeTeamPlayers": []}) is None


def test_fetch_preserves_retrieval_time_as_teacher_metadata(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "playerOfTheMatch": {
                    "value": "8.2",
                    "label": "rating",
                    "player": {"id": 7, "name": "A Player"},
                }
            }

    class FakeSession:
        def get(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(
        "src.research.sofascore_official_potm._now_utc",
        lambda: pd.Timestamp("2026-10-01T12:00:00Z"),
    )
    result = fetch_official_potm("999", session=FakeSession(), retries=1)
    assert result["status"] == "FOUND"
    assert result["label_type"] == "SOFASCORE_OFFICIAL_POTM"
    assert result["player_id"] == "7"
    assert result["label_retrieved_at_utc"] == "2026-10-01T12:00:00+00:00"
    assert result["historical_publication_time_verified"] is False


def test_attach_labels_fails_on_duplicate_event_ids():
    events = pd.DataFrame({"sofascore_event_id": ["1", "1"]})
    try:
        attach_labels(events)
    except ValueError as exc:
        assert "duplicates" in str(exc)
    else:
        raise AssertionError("duplicate event ids must fail closed")


def test_fetch_retries_only_bounded_transient_http_failures(monkeypatch):
    calls = {"n": 0}

    class FakeResponse:
        def raise_for_status(self):
            calls["n"] += 1
            if calls["n"] < 3:
                raise requests.HTTPError("temporary")
        def json(self):
            return {"playerOfTheMatch": {"player": {"id": 1, "name": "P"}}}

    class FakeSession:
        def get(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(
        "src.research.sofascore_official_potm.time.sleep",
        lambda *_: None,
    )
    result = fetch_official_potm("1", session=FakeSession(), retries=3)
    assert result["status"] == "FOUND"
    assert calls["n"] == 3


def test_attach_labels_rejects_teacher_retrieved_before_kickoff(monkeypatch):
    events = pd.DataFrame({
        "sofascore_event_id": ["1"],
        "kickoff_utc": ["2026-10-01T12:00:00Z"],
    })
    monkeypatch.setattr(
        "src.research.sofascore_official_potm.fetch_official_potm",
        lambda *args, **kwargs: {
            "event_id": "1",
            "label_type": "SOFASCORE_OFFICIAL_POTM",
            "player_id": "7",
            "player_name": "A Player",
            "rating_value": 8.0,
            "rating_label": "rating",
            "label_retrieved_at_utc": "2026-10-01T11:59:00Z",
            "source_url": "x",
            "status": "FOUND",
            "historical_publication_time_verified": False,
        },
    )
    try:
        attach_labels(events)
    except RuntimeError as exc:
        assert "before kickoff" in str(exc)
    else:
        raise AssertionError("pre-kickoff teacher retrieval must fail closed")
