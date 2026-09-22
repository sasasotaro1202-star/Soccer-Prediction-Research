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
        name="X (Twitter)",
        kind="social_api",
        role="official_club_player_journalist_pre_match_updates",
        fields=("announcements", "availability", "injuries", "suspensions", "lineup_signals", "training_signals"),
        historical=True,
        pit_capable=False,
        live_capable=True,
        auth_required=True,
        primary_for=("social_context",),
        notes="Candidate context only. Require post created_at/publication timestamp, edit-history handling, provenance, author trust tier and PIT snapshot evidence; never infer historical availability from a current profile.",
    ),
    SourceSpec(
        name="Instagram",
        kind="social_api",
        role="official_club_player_pre_match_updates",
        fields=("announcements", "availability", "injuries", "training_signals", "travel_signals", "lineup_signals"),
        historical=False,
        pit_capable=False,
        live_capable=True,
        auth_required=True,
        primary_for=("social_context",),
        notes="Professional-account API access is constrained. Treat posts/reels as research candidates until historical publication and edit/deletion provenance can be demonstrated.",
    ),
    SourceSpec(
        name="Facebook",
        kind="social_api",
        role="official_club_league_pre_match_updates",
        fields=("announcements", "availability", "injuries", "squad_context", "training_signals"),
        historical=False,
        pit_capable=False,
        live_capable=True,
        auth_required=True,
        primary_for=("social_context",),
        notes="Official Page/Graph access only. Treat post publication/backdating/editing and historical retrieval as PIT-critical; no production use until verified.",
    ),
    SourceSpec(
        name="ewalldo/Japan-J1-League-Data-and-Data-Analysis (versioned J1 2020 snapshot)",
        kind="github_versioned_dataset",
        role="j1_2020_historical_pit_evidence",
        fields=("fixtures", "results", "scores"),
        historical=True,
        pit_capable=True,
        live_capable=False,
        auth_required=False,
        primary_for=("J1",),
        notes="Audit-only third-party dataset. PIT is accepted only from an immutable commit timestamp at/after the conservative result lower bound with unique exact result identity; no current HEAD inference.",
    ),
    SourceSpec(
        name="footballcsv/cache.footballdata (weekly versioned Big-5 + Eredivisie 2019-20/2020-21 snapshots)",
        kind="github_versioned_dataset",
        role="domestic_historical_pit_evidence",
        fields=("fixtures", "results", "scores"),
        historical=True,
        pit_capable=True,
        live_capable=False,
        auth_required=False,
        primary_for=("EPL", "BL1", "LL", "FL1", "SA", "ERE"),
        notes="Audit-only Git mirror with path-level immutable snapshots. Accept only commit timestamps at/after the conservative result lower bound and unique exact date/team/score identity. Independently inspected 2017-18 and 2018-19 snapshots are observed at 2020-06-11 and may support only later prediction cutoffs; 2019-20 and 2020-21 snapshots retain their observed update windows.",
    ),
    SourceSpec(
        name="footballcsv/cache.footballdata (versioned Italy 2019-20/2020-21 snapshots)",
        kind="github_versioned_dataset",
        role="sa_historical_pit_evidence",
        fields=("fixtures", "results", "scores"),
        historical=True,
        pit_capable=True,
        live_capable=False,
        auth_required=False,
        primary_for=("SA",),
        notes="Audit-only independent Git mirror of Football-Data records. Accept only immutable commit timestamps at/after the conservative result lower bound with unique exact date/team/score identity.",
    ),
    SourceSpec(
        name="footballcsv/espana (versioned 2019-20/2020-21 snapshots)",
        kind="github_versioned_dataset",
        role="ll_historical_pit_evidence",
        fields=("fixtures", "results", "scores"),
        historical=True,
        pit_capable=True,
        live_capable=False,
        auth_required=False,
        primary_for=("LL",),
        notes="Audit-only independent Git snapshots. Accept only immutable commit timestamps at/after the conservative result lower bound with unique exact date/team/score identity.",
    ),
    SourceSpec(
        name="J.League Data Site",
        kind="official_web",
        role="j1_j2_j3_fixtures_results_tables_players",
        fields=("fixtures", "results", "tables", "player_records", "manager_records"),
        historical=True,
        pit_capable=False,
        live_capable=True,
        auth_required=False,
        primary_for=("J1", "J2", "J3"),
        notes="Official source for Japanese domestic leagues; historical publication timing is not assumed from current pages.",
    ),
    SourceSpec(
        name="Olympics / Asian Games",
        kind="official_web",
        role="asian_games_football_results_schedule_standings",
        fields=("fixtures", "results", "standings", "tournament_context"),
        historical=True,
        pit_capable=False,
        live_capable=True,
        auth_required=False,
        primary_for=("AG_M", "AG_W"),
        notes="Primary tournament record; current page reachability is not PIT proof.",
    ),
    SourceSpec(
        name="FIFA",
        kind="official_web",
        role="international_youth_tournament_results_schedule",
        fields=("fixtures", "results", "standings", "tournament_context"),
        historical=True,
        pit_capable=False,
        live_capable=True,
        auth_required=False,
        primary_for=("U23_M", "U18_M"),
        notes="Primary international competition source; age-group coverage must be verified per tournament.",
    ),
    SourceSpec(
        name="openfootball",
        kind="open_dataset",
        role="uefa_competition_historical_results",
        fields=("fixtures", "results", "scores", "competition_context"),
        historical=True,
        pit_capable=False,
        live_capable=True,
        auth_required=False,
        primary_for=("UCL", "UEL"),
        notes="Free public-domain historical results; publication-time evidence still required for strict PIT.",
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
