"""External football-data source registry.

This module deliberately separates source availability from production eligibility.
Every source must pass coverage, timestamp/PIT, leakage, schema, and chronological
OOS checks before it can influence production predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PITStatus = Literal["UNVERIFIED", "AUDIT_ONLY", "VERIFIED"]


@dataclass(frozen=True)
class ExternalSourceSpec:
    key: str
    name: str
    acquisition: str
    api_key_env: str | None
    free_tier: str
    pit_status: PITStatus
    production_enabled: bool
    priority: int
    notes: str


EXTERNAL_SOURCES: tuple[ExternalSourceSpec, ...] = (
    ExternalSourceSpec(
        key="clubelo",
        name="ClubElo",
        acquisition="public HTTP API / dated historical ratings",
        api_key_env=None,
        free_tier="public/free",
        pit_status="UNVERIFIED",
        production_enabled=False,
        priority=1,
        notes="High-priority candidate for date-aligned team-strength features.",
    ),
    ExternalSourceSpec(
        key="statsbomb_open_data",
        name="StatsBomb Open Data",
        acquisition="public GitHub JSON files",
        api_key_env=None,
        free_tier="free/open research data",
        pit_status="UNVERIFIED",
        production_enabled=False,
        priority=2,
        notes="High-value event/xG source where competition and season coverage exists.",
    ),
    ExternalSourceSpec(
        key="understat",
        name="Understat",
        acquisition="public web data extraction",
        api_key_env=None,
        free_tier="public web access",
        pit_status="UNVERIFIED",
        production_enabled=False,
        priority=3,
        notes="xG candidate; availability, timestamp reconstruction and usage constraints must be audited.",
    ),
    ExternalSourceSpec(
        key="open_meteo",
        name="Open-Meteo",
        acquisition="public weather API",
        api_key_env=None,
        free_tier="free API",
        pit_status="UNVERIFIED",
        production_enabled=False,
        priority=4,
        notes="Weather candidate; only pre-kickoff forecast/observations available by cutoff may be used.",
    ),
    ExternalSourceSpec(
        key="sofascore",
        name="SofaScore",
        acquisition="public web/API-style endpoints",
        api_key_env=None,
        free_tier="public access; verify usage limits",
        pit_status="UNVERIFIED",
        production_enabled=False,
        priority=5,
        notes="Potential lineup/event/odds supplement; strict PIT reconstruction required.",
    ),
    ExternalSourceSpec(
        key="api_football",
        name="API-Football",
        acquisition="REST API",
        api_key_env="API_FOOTBALL_KEY",
        free_tier="free tier; quota and historical availability must be verified",
        pit_status="UNVERIFIED",
        production_enabled=False,
        priority=6,
        notes="Candidate for structured fixtures, lineups, injuries and odds; never depend on a paid-only endpoint without an explicit gate.",
    ),
    ExternalSourceSpec(
        key="fbref",
        name="FBref",
        acquisition="public web pages",
        api_key_env=None,
        free_tier="public web access",
        pit_status="UNVERIFIED",
        production_enabled=False,
        priority=7,
        notes="Supplementary team/player statistics; historical point-in-time availability must be demonstrated.",
    ),
    ExternalSourceSpec(
        key="sportmonks",
        name="Sportmonks",
        acquisition="REST API",
        api_key_env="SPORTMONKS_API_TOKEN",
        free_tier="free plan; coverage/quota must be verified",
        pit_status="UNVERIFIED",
        production_enabled=False,
        priority=8,
        notes="Structured fallback/supplement; do not duplicate data already covered by stronger sources without incremental OOS evidence.",
    ),
)


def source_spec(key: str) -> ExternalSourceSpec:
    """Return a source specification by stable key."""
    for spec in EXTERNAL_SOURCES:
        if spec.key == key:
            return spec
    raise KeyError(f"unknown external source: {key}")


def production_sources() -> tuple[ExternalSourceSpec, ...]:
    """Return only explicitly production-enabled sources.

    The default is intentionally empty until PIT and OOS adoption gates promote a
    source. This prevents adding a data source from silently changing predictions.
    """
    return tuple(spec for spec in EXTERNAL_SOURCES if spec.production_enabled and spec.pit_status == "VERIFIED")


def audit_sources() -> tuple[ExternalSourceSpec, ...]:
    """Return the complete external-source audit queue in priority order."""
    return tuple(sorted(EXTERNAL_SOURCES, key=lambda spec: spec.priority))
