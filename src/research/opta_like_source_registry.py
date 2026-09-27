"""Research registry for free/public football data sources resembling Opta.

This module is metadata-only: it does not download or promote any source.
Every source is classified by information value, PIT risk and operational
constraints so downstream experiments can choose sources without weakening
the project's fail-closed rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class OptaLikeSource:
    key: str
    name: str
    kind: str
    source_url: str
    capabilities: tuple[str, ...]
    target_axes: tuple[str, ...]
    access: str
    pit_status: str
    license_status: str
    prediction_value: str
    priority: int
    notes: str = ""


_CORE_SOURCES = (
    OptaLikeSource(
        "statsbomb_open", "StatsBomb Open Data", "event+360",
        "https://github.com/hudl/open-data",
        (
            "events", "passes", "shots", "pressure", "duels", "carrying",
            "goalkeeper_actions", "lineups", "formations", "shot_locations",
            "360_freeze_frames",
        ),
        ("team_strength", "player_contribution", "tactics", "shot_quality", "defence"),
        "FREE_PUBLIC_RESEARCH", "PIT_REQUIRES_AUDIT", "SOURCE_ATTRIBUTION_REQUIRED",
        "VERY_HIGH", 100,
        "Selected competitions/matches only; 360 coverage is a subset, so absence must not become zero.",
    ),
    OptaLikeSource(
        "skillcorner_open", "SkillCorner Open Data", "tracking+dynamic_events",
        "https://github.com/SkillCorner/opendata",
        (
            "player_tracking", "ball_tracking", "dynamic_events",
            "physical_aggregates", "off_ball_runs", "passing",
            "phases_of_play", "speeds", "distances",
        ),
        ("tactics", "physical_state", "off_ball", "pressing", "transition"),
        "FREE_PUBLIC_SAMPLE", "PIT_REQUIRES_AUDIT", "CREDIT_REQUESTED",
        "VERY_HIGH", 99,
        "Ten broadcast-tracking matches plus derived dynamic events; excellent feature-engineering benchmark but limited coverage.",
    ),
    OptaLikeSource(
        "wyscout_public", "Wyscout public match-event dataset", "event",
        "https://github.com/koenvo/wyscout-soccer-match-event-dataset",
        (
            "passes", "shots", "duels", "fouls", "spatio_temporal_events",
            "players", "teams", "matches",
        ),
        ("team_strength", "player_contribution", "event_sequences", "tactics"),
        "FREE_PUBLIC_DATASET", "PIT_REQUIRES_AUDIT", "CC_BY_4_0",
        "HIGH", 96,
        "Strong event-sequence benchmark across seven competitions; historical/publication-time semantics still need verification.",
    ),
    OptaLikeSource(
        "gfdl", "Global Football (Soccer) Data Lake", "match+player_stats",
        "https://huggingface.co/datasets/eatpizzanot/soccer-dataset",
        (
            "xg", "possession", "shots", "pass_accuracy", "fouls",
            "offsides", "box_shots", "blocked_shots", "player_stats", "known_at",
        ),
        ("team_strength", "player_contribution", "shot_quality"),
        "FREE_PUBLIC_DATASET", "KNOWN_AT_AVAILABLE", "DATASET_LICENSE_AUDIT_REQUIRED",
        "HIGH", 95,
        "Already integrated research-only; known_at is the current conservative PIT boundary.",
    ),
    OptaLikeSource(
        "metrica_sample", "Metrica Sports Sample Data", "tracking+event",
        "https://github.com/metrica-sports/sample-data",
        (
            "synchronized_tracking", "events", "player_coordinates",
            "ball_coordinates", "speeds", "accelerations",
        ),
        ("tactics", "physical_state", "pitch_control", "epv"),
        "FREE_PUBLIC_SAMPLE", "PIT_NOT_PROVEN", "SAMPLE_DATA_LICENSE_AUDIT_REQUIRED",
        "HIGH", 92,
        "Small anonymized sample, primarily useful for validating tracking feature computation and leakage tests.",
    ),
    OptaLikeSource(
        "soccernet_tracking", "SoccerNet Tracking", "video_tracking",
        "https://github.com/SoccerNet/sn-tracking",
        (
            "player_tracks", "referee_tracks", "ball_tracks",
            "frame_level_locations", "tracking_benchmarks",
        ),
        ("tactics", "tracking", "video_derived_features"),
        "FREE_RESEARCH_DATASET", "PIT_NOT_PROVEN", "RESEARCH_TERMS_AUDIT_REQUIRED",
        "HIGH", 90,
        "Useful for training/benchmarking video-to-tracking components rather than direct historical team strength.",
    ),
    OptaLikeSource(
        "fbref_opta", "FBref / Stats Perform Opta-derived public tables", "public_tables",
        "https://fbref.com/",
        (
            "standard_stats", "advanced_stats", "shooting", "passing",
            "defensive_actions", "possession", "keeper", "match_logs",
        ),
        ("team_strength", "player_contribution", "role_state", "availability"),
        "PUBLIC_WEB", "HISTORICAL_PIT_UNCONFIRMED", "TERMS_AND_ATTRIBUTION_AUDIT",
        "HIGH", 88,
        "High breadth and Opta-derived statistics, but scraped publication timing is not automatically PIT-safe.",
    ),
    OptaLikeSource(
        "understat", "Understat", "shot_model",
        "https://understat.com/",
        (
            "shot_locations", "shot_xg", "xg_chain", "xg_buildup",
            "player_xg", "player_xa",
        ),
        ("shot_quality", "player_contribution", "attack_quality"),
        "PUBLIC_WEB", "HISTORICAL_PIT_UNCONFIRMED", "TERMS_AND_ACCESS_AUDIT",
        "HIGH", 87,
        "Useful independent xG view; do not merge post-match xG into a pre-kickoff row without a verified availability boundary.",
    ),
    OptaLikeSource(
        "transfermarkt", "Transfermarkt", "player_market+availability",
        "https://www.transfermarkt.com/",
        (
            "injuries", "suspensions", "transfers", "squad_values",
            "contract_context", "player_availability",
        ),
        ("availability", "squad_strength", "roster_change"),
        "PUBLIC_WEB", "HISTORICAL_PIT_UNCONFIRMED", "TERMS_AND_ACCESS_AUDIT",
        "MEDIUM_HIGH", 82,
        "Potentially valuable for roster shocks; historical availability timestamps are a major PIT risk.",
    ),
    OptaLikeSource(
        "fotmob_public", "FotMob public match data", "matchday+player",
        "https://www.fotmob.com/",
        (
            "lineups", "player_ratings", "xg", "shots", "passes",
            "possession", "live_match_state",
        ),
        ("matchday_state", "player_state", "shot_quality"),
        "PUBLIC_WEB", "CURRENT_ONLY_UNLESS_PROVEN", "TERMS_AND_ACCESS_AUDIT",
        "MEDIUM_HIGH", 80,
        "Useful as a current-match cross-check/discovery source; prior history must remain fail-closed until publication timing is proven.",
    ),
    OptaLikeSource(
        "sofascore_public", "SofaScore public match data", "matchday+event",
        "https://www.sofascore.com/",
        (
            "fixtures", "lineups", "player_stats", "incidents", "ratings",
            "odds", "match_state", "tournament_metadata",
        ),
        ("matchday_state", "availability", "market", "scope_discovery"),
        "PUBLIC_WEB", "CURRENT_SNAPSHOT_SEMANTICS", "TERMS_AND_ACCESS_AUDIT",
        "HIGH", 79,
        "Already used for discovery/matchday; retrieval time is only a conservative lower bound, not historical publication time.",
    ),
    OptaLikeSource(
        "espn_public", "ESPN public soccer data", "matchday+team",
        "https://site.api.espn.com/apis/site/v2/sports/soccer/",
        (
            "fixtures", "team_schedules", "injuries", "rosters", "odds",
            "match_summary", "competition_metadata",
        ),
        ("matchday_state", "availability", "rest", "market"),
        "FREE_KEYLESS", "CURRENT_SNAPSHOT_SEMANTICS", "PUBLIC_ENDPOINT_USAGE_AUDIT",
        "HIGH", 77,
        "Already used; strong current-match context but historical publication timing is not assumed.",
    ),
    OptaLikeSource(
        "openfootball", "openfootball", "results+fixtures",
        "https://github.com/openfootball",
        ("fixtures", "results", "competitions"),
        ("team_strength", "schedule", "scope_discovery"),
        "FREE_PUBLIC_DATA", "PIT_UNPROVEN", "REPOSITORY_LICENSE_AUDIT",
        "MEDIUM", 72,
        "Useful broad low-cost coverage and fallback, but limited advanced event detail.",
    ),
    OptaLikeSource(
        "socceraction", "socceraction", "analytics_engine",
        "https://github.com/ML-KULeuven/socceraction",
        (
            "spadl", "atomic_spadl", "xT", "VAEP", "atomic_VAEP",
            "provider_loaders", "event_value",
        ),
        ("action_value", "event_sequences", "player_contribution"),
        "FREE_OPEN_SOURCE", "N_A_LIBRARY", "MIT",
        "VERY_HIGH", 98,
        "Feature engine rather than an independent source; supports StatsBomb/Opta/Wyscout/Stats Perform/WhoScored event streams.",
    ),
    OptaLikeSource(
        "kloppy", "kloppy", "normalization_engine",
        "https://github.com/PySport/kloppy",
        (
            "event_normalization", "tracking_normalization", "coordinate_conversion",
            "provider_adapters", "event_queries", "dataframes",
        ),
        ("data_integration", "tracking", "event_sequences"),
        "FREE_OPEN_SOURCE", "N_A_LIBRARY", "BSD_3_CLAUSE",
        "VERY_HIGH", 97,
        "Unifies provider-specific event/tracking schemas and reduces integration risk as source count grows.",
    ),
    OptaLikeSource(
        "laurie_on_tracking", "LaurieOnTracking", "tracking_feature_engine",
        "https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking",
        (
            "velocity", "acceleration", "pitch_control", "EPV",
            "formation", "passing_options",
        ),
        ("tactics", "physical_state", "pitch_control", "epv", "decision_quality"),
        "FREE_OPEN_SOURCE", "N_A_LIBRARY", "MIT",
        "VERY_HIGH", 93,
        "Reference implementation for deriving higher-order spatial features from tracking data.",
    ),
    OptaLikeSource(
        "worldfootballR_data", "worldfootballR data", "pre_scraped_public_tables",
        "https://github.com/JaseZiv/worldfootballR_data",
        (
            "fbref_stats", "fbref_results", "understat_shots",
            "transfermarkt_data",
        ),
        ("team_strength", "player_contribution", "shot_quality", "availability"),
        "FREE_PUBLIC_DATASET", "HISTORICAL_PIT_UNCONFIRMED", "SOURCE_SPECIFIC_TERMS_AUDIT",
        "HIGH", 84,
        "Convenient pre-collected data, but archival publication timing is not itself proof of prediction-time availability.",
    ),
)


_EXTENDED_SOURCES = (
    OptaLikeSource("football_data_org", "football-data.org", "fixtures+standings", "https://www.football-data.org/", ("fixtures","schedules","standings","competition_metadata"), ("schedule","team_strength","scope_discovery"), "FREE_PLAN", "CURRENT_FREE_ACCESS", "FREE_PLAN_TERMS_AUDIT", "MEDIUM", 73, "Free plan exists but delayed scores/schedules and limited competition count; advanced/deep/live features are paid."),
    OptaLikeSource("clubelo_archive", "ClubElo historical snapshots", "elo_history", "https://clubelo.com/", ("elo","team_history","global_strength"), ("team_strength","regime","prior_strength"), "PUBLIC_ARCHIVE_OR_MIRROR", "CURRENT_ACCESS_UNCONFIRMED", "SOURCE_ACCESS_AUDIT", "HIGH", 74, "Useful as a strength prior when archived snapshots are available; current API access should not be assumed."),
    OptaLikeSource("fivethirtyeight_spi", "FiveThirtyEight SPI historical data", "model_reference", "https://github.com/fivethirtyeight/data/tree/master/soccer-spi", ("spi","pre_match_probability","projected_score","xg","importance","strength"), ("team_strength","benchmark","calibration_reference"), "FREE_ARCHIVED_DATA", "HISTORICAL_ONLY", "CC_BY_4_0", "HIGH", 71, "Historical SPI data are valuable as an independent benchmark/reference; FiveThirtyEight states sports forecasts stopped updating in 2023."),

    OptaLikeSource("driblab_open", "Driblab Open Data", "tracking", "https://github.com/driblab/open-data", ("player_tracking","ball_tracking","player_metadata","positions","velocity","acceleration"), ("tactics","physical_state","off_ball","transition"), "FREE_PUBLIC_SAMPLE", "PIT_NOT_PROVEN", "REPOSITORY_TERMS_AUDIT", "VERY_HIGH", 94, "Ten 10-FPS broadcast-tracking matches from major European competitions; strong cross-league spatial benchmark."),
    OptaLikeSource("idsse_dfl_open", "IDSSE / DFL open synchronized data", "tracking+event", "https://github.com/PySport/kloppy", ("tracab_tracking","synchronized_event_data","player_positions","ball_positions","bundesliga_matches"), ("tactics","tracking","event_sequences","pitch_control"), "FREE_PUBLIC_DATASET", "PIT_NOT_PROVEN", "CC_BY_4_0", "VERY_HIGH", 91, "Open synchronized TRACAB tracking plus DFL event data; underlying dataset terms still require audit."),
    OptaLikeSource("databallpy", "DataBallPy", "tracking_event_engine", "https://github.com/Alek050/databallpy", ("tracking_event_sync","coordinate_processing","provider_loaders","sportec_loader","dfl_open_games"), ("data_integration","tracking","event_sequences"), "FREE_OPEN_SOURCE", "N_A_LIBRARY", "OPEN_SOURCE_LICENSE_AUDIT", "HIGH", 89, "Supports Metrica, Inmotio, TRACAB, DFL/Sportec, Opta/StatsPerform, StatsBomb and Wyscout formats."),
    OptaLikeSource("open_starlab_preprocessing", "open-starlab football preprocessing", "preprocessing_engine", "https://github.com/open-starlab/PreProcessing", ("event_preprocessing","unified_event_format","nmpstpp","statsbomb","wyscout","sportec","metrica","soccertrackv2"), ("data_integration","event_sequences","forecasting"), "FREE_OPEN_SOURCE", "N_A_LIBRARY", "OPEN_SOURCE_LICENSE_AUDIT", "HIGH", 86, "Compatibility layer for many provider formats; underlying provider rights still apply."),
    OptaLikeSource("soccermatics", "Soccermatics", "tracking_event_feature_engine", "https://github.com/JoGall/soccermatics", ("tracking_visualization","shot_maps","average_positions","heatmaps","player_trajectories","tracking_helpers"), ("tactics","spatial_features","player_state"), "FREE_OPEN_SOURCE", "N_A_LIBRARY", "OPEN_SOURCE_LICENSE_AUDIT", "MEDIUM_HIGH", 83, "Spatial feature utilities; useful for representation and diagnostics rather than as an independent data feed."),
    OptaLikeSource("football_data_xg_reference", "StatsBomb xG reference implementation", "xg_feature_engine", "https://github.com/carrba/football-data-xg", ("shot_xg","shot_context","xg_training_pipeline"), ("shot_quality","model_validation"), "FREE_OPEN_SOURCE", "N_A_LIBRARY", "REPOSITORY_LICENSE_AUDIT", "MEDIUM_HIGH", 81, "Reference implementation for independent xG feature experiments using StatsBomb open data."),
    OptaLikeSource("last_row_tracking", "Last Row tracking sample", "tracking", "https://github.com/Friends-of-Tracking-Data-FoTD/Last-Row", ("player_tracking","ball_tracking","goal_sequences"), ("tactics","transition","tracking"), "FREE_PUBLIC_SAMPLE", "PIT_NOT_PROVEN", "SOURCE_CREDIT_REQUIRED", "MEDIUM", 76, "Small 2D tracking sample of Liverpool goal sequences; useful for validating spatial feature code."),
    OptaLikeSource("hammarby_signality", "Signality / Hammarby tracking sample", "tracking", "https://uppsala.instructure.com/courses/28112/pages/4-player-movements-on-the-pitch", ("player_movements","tracking"), ("tactics","tracking","physical_state"), "FREE_RESEARCH_SAMPLE", "PIT_NOT_PROVEN", "COURSE_ACCESS_TERMS_AUDIT", "MEDIUM", 75, "Three Hammarby matches referenced by Metrica's sample-data documentation."),
    OptaLikeSource("pff_fc_2022", "PFF FC 2022 World Cup data", "tracking+event", "https://github.com/PySport/kloppy", ("broadcast_tracking","event_data","play_by_play_grades"), ("tactics","tracking","player_contribution"), "FREE_ACCESS_REQUEST", "PIT_NOT_PROVEN", "ACCESS_REQUEST_TERMS_AUDIT", "HIGH", 78, "Reported as covering all 64 matches of the 2022 men's World Cup; request-based access remains optional."),
)
_SOURCES = _CORE_SOURCES + _EXTENDED_SOURCES


def all_sources() -> tuple[OptaLikeSource, ...]:
    return _SOURCES


def usable_source_candidates(*, target_axis: str, include_unproven_pit: bool = True) -> tuple[OptaLikeSource, ...]:
    """Return candidates sorted by research priority without asserting usability."""
    candidates = [
        source for source in _SOURCES
        if target_axis in source.target_axes
        and (include_unproven_pit or source.pit_status in {"KNOWN_AT_AVAILABLE", "CURRENT_SNAPSHOT_SEMANTICS"})
    ]
    return tuple(sorted(candidates, key=lambda s: (-s.priority, s.key)))


def data_sources_only() -> tuple[OptaLikeSource, ...]:
    return tuple(source for source in _SOURCES if not source.kind.endswith("_engine"))


def source_summary() -> dict[str, int]:
    return {
        "total": len(_SOURCES),
        "data_sources": len(data_sources_only()),
        "free_or_public": sum(
            source.access.startswith(("FREE_", "PUBLIC_")) or source.access == "FREE_PLAN" for source in _SOURCES
        ),
        "pit_known_or_current": sum(
            source.pit_status in {"KNOWN_AT_AVAILABLE", "CURRENT_SNAPSHOT_SEMANTICS"} for source in _SOURCES
        ),
        "research_only_high_value": sum(
            source.prediction_value in {"VERY_HIGH", "HIGH"}
            for source in _SOURCES
        ),
    }


def source_keys(sources: Iterable[OptaLikeSource] | None = None) -> tuple[str, ...]:
    values = tuple(sources) if sources is not None else _SOURCES
    return tuple(source.key for source in values)
