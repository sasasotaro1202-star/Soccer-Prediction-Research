from src.data.social_pit import normalize_social_evidence


CUTOFF = "2026-09-10T12:00:00Z"
RETRIEVED = "2026-09-10T12:05:00Z"


def test_x_post_with_explicit_created_at_is_pit_safe():
    x = {
        "data": {
            "id": "1",
            "author_id": "club-1",
            "created_at": "2026-09-10T10:00:00Z",
            "text": "Training update",
        }
    }
    r = normalize_social_evidence("x_api", x, retrieved_at_utc=RETRIEVED, prediction_cutoff_at_utc=CUTOFF)
    assert r.pit_safe is True
    assert r.pit_reason == "explicit_publication_time_before_cutoff"
    assert r.raw_sha256


def test_missing_publication_time_is_fail_closed():
    x = {"data": {"id": "2", "author_id": "club-1", "text": "Unknown timing"}}
    r = normalize_social_evidence("x_api", x, retrieved_at_utc=RETRIEVED, prediction_cutoff_at_utc=CUTOFF)
    assert r.pit_safe is False
    assert r.pit_reason == "publication_time_missing_or_invalid"


def test_post_after_cutoff_is_not_usable():
    x = {
        "data": {
            "id": "3",
            "author_id": "club-1",
            "created_at": "2026-09-10T13:00:00Z",
            "text": "Late update",
        }
    }
    r = normalize_social_evidence("x_api", x, retrieved_at_utc=RETRIEVED, prediction_cutoff_at_utc=CUTOFF)
    assert r.pit_safe is False
    assert r.pit_reason == "published_after_prediction_cutoff"


def test_post_edited_after_cutoff_is_not_usable():
    x = {
        "data": {
            "id": "4",
            "author_id": "club-1",
            "created_at": "2026-09-10T10:00:00Z",
            "edited_at": "2026-09-10T13:00:00Z",
            "text": "Edited update",
        }
    }
    r = normalize_social_evidence("x_api", x, retrieved_at_utc=RETRIEVED, prediction_cutoff_at_utc=CUTOFF)
    assert r.pit_safe is False
    assert r.pit_reason == "content_edited_after_prediction_cutoff"
