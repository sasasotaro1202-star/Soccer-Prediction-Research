from __future__ import annotations

"""Central registry for soccer data providers and their admissible research roles.

The registry is metadata only: it never converts missing data to zero and never
assumes that a provider's advertised coverage is actual row-level coverage.
Actual acquisition is measured by adapters and recorded in the coverage matrix.
"""

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class SourceSpec:
    name: str
    kind: str
    role: str
    fields: tuple[str, ...]
    historical: bool
    pit_capable: bool
    live_capable: bool
    auth_required: bool
    primary_for: tuple[str, ...] = ()
    notes: str = ""


SOCCER_SOURCES: Final[tuple[SourceSpec, ...]] = (
    SourceSpec(
        name="football-data.co.uk",
        kind="csv",
        role="historical_match_results_stats_odds",
        fields=("fixtures", "result", "score", "basic_stats", "odds"),
        historical=True,
        pit_capable=True,
        live_capable=False,
        auth_required=False,
        primary_for=("historical_baseline",),
        notes="Static season CSVs; archive evidence required for strict PIT.",
    ),
    SourceSpec(
        name="API-Football",
        kind="api",
        role="fixtures_events_lineups_stats_players_injuries_odds",
        fields=("fixtures", "events", "lineups", "statistics", "players", "injuries", "odds", "standings"),
        historical=True,
        pit_capable=False,
        live_capable=True,
        auth_required=True,
        primary_for=("production", "pre_match_context"),
        notes="Coverage is league-season-field specific and must be queried before acquisition.",
    ),
    SourceSpec(
        name="Sofascore",
        kind="public_api",
        role="events_lineups_stats_incidents_players_ratings",
        fields=("fixtures", "events", "lineups", "statistics", "player_stats", "ratings", "incidents"),
        historical=True,
        pit_capable=False,
        live_capable=True,
        auth_required=False,
        primary_for=("match_detail", "player_context"),
        notes="Public endpoints may change; snapshot raw responses and timestamps.",
    ),
    SourceSpec(
        name="FotMob",
        kind="public_web_api",
        role="fixtures_xg_lineups_player_ratings_match_stats",
        fields=("fixtures", "xg", "lineups", "player_stats", "ratings", "match_stats"),
        historical=True,
        pit_capable=False,
        live_capable=True,
        auth_required=False,
        primary_for=("xg", "player_context"),
        notes="Treat endpoint/schema stability as a QC dimension; archive raw snapshots.",
    ),
    SourceSpec(
        name="Understat",
        kind="public_web",
        role="xg_xga_shots_player_xg",
        fields=("xg", "xga", "shots", "player_xg"),
        historical=True,
        pit_capable=False,
        live_capable=False,
        auth_required=False,
        primary_for=("xg_history",),
        notes="Historical xG source; acquisition and entity mapping must be audited.",
    ),
    SourceSpec(
        name="FBref / Stats",
        kind="public_web",
        role="team_player_advanced_stats",
        fields=("team_stats", "player_stats", "shooting", "passing", "defense", "possession", "keeper"),
        historical=True,
        pit_capable=False,
        live_capable=False,
        auth_required=False,
        primary_for=("advanced_team_stats", "player_history"),
        notes="Use only fields whose publication timing can be established for PIT research.",
    ),
    SourceSpec(
        name="StatsBomb Open Data",
        kind="open_dataset",
        role="event_level_match_data",
        fields=("events", "shots", "passes", "pressures", "lineups", "player_events"),
        historical=True,
        pit_capable=False,
        live_capable=False,
        auth_required=False,
        primary_for=("event_features",),
        notes="Competition coverage is selective; never assume all leagues/seasons exist.",
    ),
    SourceSpec(
        name="Transfermarkt",
        kind="public_web",
        role="squad_transfers_injuries_market_context",
        fields=("squads", "transfers", "injuries", "market_values", "managers"),
        historical=True,
        pit_capable=False,
        live_capable=False,
        auth_required=False,
        primary_for=("squad_context",),
        notes="Publication timestamps and historical snapshots must be verified before PIT use.",
    ),
    SourceSpec(
        name="WorldFootball.net",
        kind="public_web",
        role="fixtures_results_tables_historical_context",
        fields=("fixtures", "results", "tables", "competition_context"),
        historical=True,
        pit_capable=False,
        live_capable=False,
        auth_required=False,
        primary_for=("cross_check",),
        notes="Independent reconciliation source; not automatically treated as PIT evidence.",
    ),
    SourceSpec(
        name="Internet Archive / Wayback",
        kind="archive",
        role="historical_snapshot_evidence",
        fields=("archived_source",),
        historical=True,
        pit_capable=True,
        live_capable=False,
        auth_required=False,
        primary_for=("pit_evidence",),
        notes="Capture timestamp is evidence of an archived observation, not automatically original publication time.",
    ),
    SourceSpec(
        name="Arquivo.pt",
        kind="archive",
        role="historical_snapshot_evidence_secondary",
        fields=("archived_source",),
        historical=True,
        pit_capable=True,
        live_capable=False,
        auth_required=False,
        primary_for=("pit_evidence_secondary",),
        notes="Secondary archive; exact replay identity and result matching are mandatory.",
    ),
)


def get_source(name: str) -> SourceSpec:
    for source in SOCCER_SOURCES:
        if source.name == name:
            return source
    raise KeyError(f"Unknown soccer data source: {name}")


def source_names() -> tuple[str, ...]:
    return tuple(source.name for source in SOCCER_SOURCES)


def pit_sources() -> tuple[SourceSpec, ...]:
    return tuple(source for source in SOCCER_SOURCES if source.pit_capable)
