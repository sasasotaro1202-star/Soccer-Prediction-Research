"""PIT-aware adapters for external football data sources.

The adapters deliberately separate acquisition from adoption.  A source may be
fetchable without being safe for historical OOS.  Callers must inspect
``FeatureRecord.pit_safe`` before using a value in a prediction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from src.data.external_fetch import CachedResponse, ExternalFetcher, iso_utc, pit_is_safe


@dataclass(frozen=True)
class FeatureRecord:
    source: str
    entity_key: str
    value: Any
    source_timestamp: str | None
    feature_available_at: str | None
    prediction_cutoff_at: str | None
    pit_safe: bool
    content_sha256: str


@dataclass(frozen=True)
class SourceRequest:
    source: str
    url: str
    params: dict[str, Any]
    feature_available_at: str | None = None
    source_timestamp: str | None = None


def _as_utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return iso_utc(value)


def _cutoff(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _as_utc(value)
    return str(value)


class ExternalAdapter:
    source: str

    def __init__(self, fetcher: ExternalFetcher | None = None) -> None:
        self.fetcher = fetcher or ExternalFetcher()

    def fetch(self, request: SourceRequest, *, prediction_cutoff_at: datetime | str | None = None) -> CachedResponse:
        cutoff = _cutoff(prediction_cutoff_at)
        return self.fetcher.get(
            request.source,
            request.url,
            params=request.params,
            source_timestamp=request.source_timestamp,
            feature_available_at=request.feature_available_at,
            prediction_cutoff_at=cutoff,
        )

    @staticmethod
    def feature(response: CachedResponse, entity_key: str, value: Any, *, prediction_cutoff_at: datetime | str | None = None) -> FeatureRecord:
        cutoff = _cutoff(prediction_cutoff_at)
        available = response.metadata.feature_available_at
        return FeatureRecord(
            source=response.metadata.source,
            entity_key=entity_key,
            value=value,
            source_timestamp=response.metadata.source_timestamp,
            feature_available_at=available,
            prediction_cutoff_at=cutoff,
            pit_safe=pit_is_safe(available, cutoff),
            content_sha256=response.metadata.content_sha256,
        )


class ClubEloAdapter(ExternalAdapter):
    """Date-aligned ClubElo ratings. Public endpoint; no API key."""

    source = "clubelo"
    base_url = "https://api.clubelo.com"

    def request(self, as_of: datetime) -> SourceRequest:
        stamp = _as_utc(as_of)
        date_key = as_of.astimezone(timezone.utc).strftime("%Y-%m-%d")
        return SourceRequest(self.source, f"{self.base_url}/{date_key}", {}, source_timestamp=stamp, feature_available_at=stamp)

    @staticmethod
    def parse_ratings(response: CachedResponse) -> list[dict[str, Any]]:
        import csv
        from io import StringIO

        text = response.body.decode("utf-8-sig")
        rows = list(csv.DictReader(StringIO(text)))
        return rows


class StatsBombOpenDataAdapter(ExternalAdapter):
    """StatsBomb Open Data GitHub JSON acquisition.

    Raw event data is historical and useful for pre-match rolling features only
    after the feature builder enforces an event/match cutoff.  Repository update
    timestamps are provenance, not permission to use future match events.
    """

    source = "statsbomb_open_data"
    base_url = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

    def request(self, path: str, *, feature_available_at: datetime | str | None = None) -> SourceRequest:
        available = _cutoff(feature_available_at)
        return SourceRequest(self.source, f"{self.base_url}/{path.lstrip('/')}", {}, feature_available_at=available)

    @staticmethod
    def parse_json(response: CachedResponse) -> Any:
        return json.loads(response.body.decode("utf-8"))


class OpenMeteoAdapter(ExternalAdapter):
    """Open-Meteo forecast acquisition.

    Historical OOS use is intentionally not inferred from today's forecast API.
    A caller must supply the forecast snapshot's own availability timestamp.
    """

    source = "open_meteo"
    base_url = "https://api.open-meteo.com/v1/forecast"

    def request(self, *, latitude: float, longitude: float, hourly: str, feature_available_at: datetime | str) -> SourceRequest:
        return SourceRequest(
            self.source,
            self.base_url,
            {"latitude": latitude, "longitude": longitude, "hourly": hourly},
            feature_available_at=_cutoff(feature_available_at),
        )


class OpenMeteoHistoricalForecastAdapter(ExternalAdapter):
    """Archived forecast snapshots for PIT-safe weather research.

    The caller must provide the forecast run initialisation/availability timestamp.
    We never equate a run's valid time with its publication time. The provider's
    archived single-run API is therefore used only when an explicit availability
    cutoff can be supplied and audited.
    """

    source = "open_meteo"
    base_url = "https://historical-forecast-api.open-meteo.com/v1/forecast"

    def request(
        self,
        *,
        latitude: float,
        longitude: float,
        hourly: str,
        run: datetime | str,
        feature_available_at: datetime | str,
    ) -> SourceRequest:
        return SourceRequest(
            self.source,
            self.base_url,
            {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": hourly,
                "run": _cutoff(run),
            },
            source_timestamp=_cutoff(run),
            feature_available_at=_cutoff(feature_available_at),
        )


class ConfiguredApiAdapter(ExternalAdapter):
    """Generic authenticated adapter used by API-Football/Sportmonks.

    Authentication is supplied through the process environment by the caller;
    no secret is stored in source control.  The adapter only constructs requests
    and never treats key presence as PIT verification.
    """

    def __init__(self, source: str, base_url: str, *, headers_factory: Callable[[], dict[str, str]] | None = None, fetcher: ExternalFetcher | None = None) -> None:
        super().__init__(fetcher)
        self.source = source
        self.base_url = base_url.rstrip("/")
        self.headers_factory = headers_factory or (lambda: {})

    def request(self, path: str, params: dict[str, Any] | None = None, *, feature_available_at: datetime | str | None = None) -> SourceRequest:
        return SourceRequest(self.source, f"{self.base_url}/{path.lstrip('/')}", params or {}, feature_available_at=_cutoff(feature_available_at))

    def fetch(self, request: SourceRequest, *, prediction_cutoff_at: datetime | str | None = None) -> CachedResponse:
        cutoff = _cutoff(prediction_cutoff_at)
        return self.fetcher.get(
            request.source,
            request.url,
            params=request.params,
            source_timestamp=request.source_timestamp,
            feature_available_at=request.feature_available_at,
            prediction_cutoff_at=cutoff,
            headers=self.headers_factory(),
        )


ADAPTER_FACTORIES: dict[str, Callable[[], ExternalAdapter]] = {
    "clubelo": ClubEloAdapter,
    "statsbomb_open_data": StatsBombOpenDataAdapter,
    "open_meteo": OpenMeteoAdapter,\n    "open_meteo_historical_forecast": OpenMeteoHistoricalForecastAdapter,
    "api_football": lambda: ConfiguredApiAdapter("api_football", "https://v3.football.api-sports.io"),
    "sportmonks": lambda: ConfiguredApiAdapter("sportmonks", "https://api.sportmonks.com/v3/football"),
}


def available_adapters() -> tuple[str, ...]:
    return tuple(ADAPTER_FACTORIES)
