from datetime import datetime, timezone

from src.data.external_adapters import ClubEloAdapter, OpenMeteoAdapter, StatsBombOpenDataAdapter, available_adapters
from src.data.external_fetch import CachedResponse, FetchMetadata


def test_adapter_registry_is_deterministic():
    assert available_adapters() == (
        "clubelo",
        "statsbomb_open_data",
        "open_meteo",
        "api_football",
        "sportmonks",
    )


def test_clubelo_request_is_date_aligned():
    request = ClubEloAdapter().request(datetime(2026, 9, 14, 12, tzinfo=timezone.utc))
    assert request.url.endswith("/2026-09-14")
    assert request.source == "clubelo"
    assert request.feature_available_at == "2026-09-14T12:00:00Z"


def test_statsbomb_request_never_invents_availability_time():
    request = StatsBombOpenDataAdapter().request("matches/16/4.json")
    assert request.url.endswith("/matches/16/4.json")
    assert request.feature_available_at is None


def test_open_meteo_requires_snapshot_availability_time():
    request = OpenMeteoAdapter().request(
        latitude=51.5,
        longitude=-0.12,
        hourly="temperature_2m",
        feature_available_at="2026-09-14T10:00:00Z",
    )
    assert request.feature_available_at == "2026-09-14T10:00:00Z"
    assert request.params["hourly"] == "temperature_2m"


def test_feature_metadata_keeps_raw_digest_and_pit_state():
    response = CachedResponse(
        b"{}",
        FetchMetadata(
            source="statsbomb_open_data",
            request_key="k",
            retrieved_at="2026-09-14T10:00:00Z",
            http_status=200,
            content_sha256="abc",
            content_bytes=2,
            cache_hit=False,
            feature_available_at="2026-09-14T09:00:00Z",
        ),
    )
    record = StatsBombOpenDataAdapter.feature(response, "team:Arsenal", 0.42, prediction_cutoff_at="2026-09-14T10:00:00Z")
    assert record.pit_safe is True
    assert record.content_sha256 == "abc"
    assert record.value == 0.42
