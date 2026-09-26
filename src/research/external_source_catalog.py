from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SourceClass = Literal[
    "forecast",
    "rating",
    "performance_data",
    "player_status",
    "weather",
    "fixture_data",
    "entity_data",
]


@dataclass(frozen=True)
class ExternalSourceProfile:
    key: str
    name: str
    source_class: SourceClass
    role: str
    breadth: str
    strengths: tuple[str, ...]
    pit_default: str
    production_status: str


SOURCE_CATALOG: tuple[ExternalSourceProfile, ...] = (
    ExternalSourceProfile(
        key="opta_analyst",
        name="Opta Analyst",
        source_class="forecast",
        role="External 1X2 prior / benchmark",
        breadth="Major competitions; public historical coverage is incomplete",
        strengths=("1X2 probabilities", "model-based season/match forecasts"),
        pit_default="UNKNOWN_UNLESS_DOCUMENTED",
        production_status="RESEARCH_ONLY",
    ),
    ExternalSourceProfile(
        key="forebet",
        name="Forebet",
        source_class="forecast",
        role="Broad-coverage independent probability prior",
        breadth="Very broad global coverage; Forebet site/FAQ reports 850+ to 1,200+ leagues",
        strengths=(
            "1X2 probabilities",
            "correct-score prediction",
            "goals / BTTS / handicap / cards / corners",
        ),
        pit_default="UNKNOWN_FOR_HISTORICAL_REPLAY",
        production_status="RESEARCH_ONLY",
    ),
    ExternalSourceProfile(
        key="predictz",
        name="PredictZ",
        source_class="forecast",
        role="Broad-coverage score/tip benchmark",
        breadth="Broad global league coverage",
        strengths=("1X2 odds display", "correct-score tips", "recent-form context"),
        pit_default="UNKNOWN_FOR_HISTORICAL_REPLAY",
        production_status="RESEARCH_ONLY",
    ),
    ExternalSourceProfile(
        key="oddspedia_smartbet",
        name="Oddspedia SmartBet",
        source_class="forecast",
        role="External model benchmark",
        breadth="Broad match coverage varies by competition",
        strengths=(
            "1X2 model probabilities",
            "injuries / player stats / H2H inputs",
            "continuous odds comparison",
        ),
        pit_default="UNKNOWN_FOR_HISTORICAL_REPLAY",
        production_status="RESEARCH_ONLY",
    ),
    ExternalSourceProfile(
        key="clubelo",
        name="ClubElo",
        source_class="rating",
        role="Independent team-strength prior",
        breadth="Long historical coverage across European club football",
        strengths=("historical Elo", "date-specific team strength", "home-field adjustment"),
        pit_default="AVAILABLE_AS_RATING_SNAPSHOT",
        production_status="RESEARCH_CANDIDATE",
    ),
    ExternalSourceProfile(
        key="fivethirtyeight_spi",
        name="FiveThirtyEight SPI",
        source_class="forecast",
        role="Historical external forecast benchmark",
        breadth="Published match forecasts back to 2016",
        strengths=("historical forecast archive", "pre-match SPI ratings", "replay benchmark"),
        pit_default="DATE_LEVEL_ONLY_UNLESS_TIMESTAMPED",
        production_status="RESEARCH_ONLY",
    ),
    ExternalSourceProfile(
        key="statsbomb_open",
        name="StatsBomb Open Data",
        source_class="performance_data",
        role="Event / lineup / 360 enrichment",
        breadth="Selected competitions and seasons",
        strengths=("event data", "lineups", "selected 360 data", "research reproducibility"),
        pit_default="MATCH_EVENT_TIMES_AVAILABLE_BUT_FEATURE_AVAILABILITY_MUST_BE_CHECKED",
        production_status="RESEARCH_CANDIDATE",
    ),
    ExternalSourceProfile(
        key="understat",
        name="Understat",
        source_class="performance_data",
        role="Shot-level xG enrichment",
        breadth="Big 5 leagues and RFPL in the supported public ecosystem",
        strengths=("shot locations", "xG", "player/team attacking process"),
        pit_default="DELAYED_FEATURE_SOURCE;_VERIFY_AVAILABILITY",
        production_status="RESEARCH_CANDIDATE",
    ),
    ExternalSourceProfile(
        key="fbref",
        name="FBref",
        source_class="performance_data",
        role="Team/player advanced-stat enrichment",
        breadth="Wide league/competition coverage; metric availability varies",
        strengths=("advanced team stats", "player stats", "Opta-provided advanced data on supported competitions"),
        pit_default="VERIFY_PUBLICATION_TIME",
        production_status="RESEARCH_CANDIDATE",
    ),
    ExternalSourceProfile(
        key="transfermarkt",
        name="Transfermarkt",
        source_class="player_status",
        role="Injury / suspension / transfer context",
        breadth="Very broad player/club coverage",
        strengths=("injury status", "suspensions", "transfers", "squad context"),
        pit_default="VERIFY_TIMESTAMP_AND_STATUS_EFFECTIVE_TIME",
        production_status="RESEARCH_CANDIDATE",
    ),
    ExternalSourceProfile(
        key="open_meteo",
        name="Open-Meteo",
        source_class="weather",
        role="Match-location weather and forecast-state enrichment",
        breadth="Global",
        strengths=(
            "historical weather",
            "historical forecast archives",
            "previous model runs",
            "lead-time-specific weather reconstruction",
        ),
        pit_default="FORECAST_RUN_TIMESTAMP_AVAILABLE_FOR_HISTORICAL_FORECAST_APIS",
        production_status="RESEARCH_CANDIDATE",
    ),
    ExternalSourceProfile(
        key="football_data",
        name="Football-Data.co.uk",
        source_class="fixture_data",
        role="Historical results, match statistics and bookmaker-odds benchmark",
        breadth="Multiple European and worldwide league datasets; country coverage varies",
        strengths=(
            "free CSV archives",
            "match results",
            "team/match statistics",
            "opening-period odds",
            "closing odds",
            "documented fixture-list collection times",
        ),
        pit_default="USE_DOCUMENTED_FIXTURE_COLLECTION_WINDOWS_WHEN_APPLICABLE",
        production_status="RESEARCH_CANDIDATE",
    ),
    ExternalSourceProfile(
        key="openligadb",
        name="OpenLigaDB",
        source_class="fixture_data",
        role="Free fixture/result fallback and German-league enrichment",
        breadth="Bundesliga and many additional leagues",
        strengths=("no authentication", "match data", "league/season lookup", "60 requests/minute/IP"),
        pit_default="VERIFY_RETRIEVAL_AND_UPDATE_TIME_FOR_LIVE_USE",
        production_status="RESEARCH_CANDIDATE",
    ),
    ExternalSourceProfile(
        key="api_football_free",
        name="API-Football Free Tier",
        source_class="fixture_data",
        role="Broad structured football API fallback",
        breadth="All listed competitions/endpoints, but free plan has recent-season limits",
        strengths=(
            "fixtures",
            "events",
            "lineups",
            "statistics",
            "odds",
            "predictions",
            "injuries",
            "player transfers",
        ),
        pit_default="PROVIDER_TIMESTAMP_AND_REQUEST_TIME_MUST_BE_RECORDED",
        production_status="RESEARCH_CANDIDATE_FREE_TIER_ONLY",
    ),
    ExternalSourceProfile(
        key="thesportsdb_v1",
        name="TheSportsDB v1",
        source_class="fixture_data",
        role="Low-volume fixture/lineup/player fallback and entity discovery",
        breadth="Large multi-sport database; football coverage varies by league",
        strengths=("free v1 API", "fixtures", "events", "lineups", "players", "season data"),
        pit_default="VERIFY_PROVIDER_TIMESTAMP_AND_ARCHIVAL_AVAILABILITY",
        production_status="RESEARCH_CANDIDATE",
    ),
    ExternalSourceProfile(
        key="rsssf",
        name="RSSSF",
        source_class="fixture_data",
        role="Historical/rare-competition result archive and scope discovery",
        breadth="Worldwide domestic/international competitions, including lower-profile countries",
        strengths=("deep historical archive", "country/competition indexes", "rare coverage"),
        pit_default="ARCHIVAL_RESULTS_ONLY;_NOT_A_MATCH-TIME_FEATURE_SOURCE",
        production_status="RESEARCH_CANDIDATE",
    ),
    ExternalSourceProfile(
        key="openfootball_world",
        name="OpenFootball World",
        source_class="fixture_data",
        role="Broad fixture/result and competition-scope expansion",
        breadth="North America, Asia, Africa, Australia, Europe and more",
        strengths=("public-domain fixtures", "historical results", "scope discovery"),
        pit_default="MATCH_DATE_ONLY;_FEATURE_AVAILABILITY_REQUIRES_SEPARATE_TIMESTAMP",
        production_status="RESEARCH_CANDIDATE",
    ),
    ExternalSourceProfile(
        key="openfootball_players",
        name="OpenFootball Players",
        source_class="entity_data",
        role="Player/entity normalization and discovery",
        breadth="Country-by-country player reference data",
        strengths=("player names", "positions", "date of birth", "entity matching support"),
        pit_default="REFERENCE_DATA;_NOT_A_MATCH-TIME_FEATURE",
        production_status="RESEARCH_CANDIDATE",
    ),
)


def get_source(key: str) -> ExternalSourceProfile:
    for source in SOURCE_CATALOG:
        if source.key == key:
            return source
    raise KeyError(key)


def sources_by_class(source_class: SourceClass) -> tuple[ExternalSourceProfile, ...]:
    return tuple(source for source in SOURCE_CATALOG if source.source_class == source_class)
