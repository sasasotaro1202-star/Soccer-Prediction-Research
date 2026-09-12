from __future__ import annotations

from dataclasses import dataclass

from src.data.fixture_field_audit import TARGET_COMPETITIONS


@dataclass(frozen=True)
class CompetitionSourcePlan:
    competition: str
    canonical_candidates: tuple[str, ...]
    discovery_only: tuple[str, ...]
    pit_status: str
    notes: str


# This is a plan, not a coverage claim. Actual row/field coverage must come from
# adapter execution and be written to the audit artifacts.
PLANS = {
    "EPL": ("Football-Data.co.uk", "football-data.org", "Sportmonks"),
    "CHA": ("Football-Data.co.uk", "football-data.org", "Sportmonks"),
    "BL1": ("Football-Data.co.uk", "football-data.org", "Sportmonks"),
    "SA": ("Football-Data.co.uk", "football-data.org", "Sportmonks"),
    "LL": ("Football-Data.co.uk", "football-data.org", "Sportmonks"),
    "FL1": ("Football-Data.co.uk", "football-data.org", "Sportmonks"),
    "UCL": ("football-data.org", "Sportmonks", "TheStatsAPI"),
    "UEL": ("football-data.org", "Sportmonks"),
    "J1": ("football-data.org", "Sportmonks", "TheStatsAPI"),
    "J2": ("Sportmonks", "TheStatsAPI", "J.League"),
    "J3": ("Sportmonks", "TheStatsAPI", "J.League"),
    "DFBP": ("Sportmonks",),
    "CAR": ("Sportmonks",),
    "FRI": ("Sportmonks", "ESPN"),
}


def source_plans() -> tuple[CompetitionSourcePlan, ...]:
    return tuple(
        CompetitionSourcePlan(
            competition=c,
            canonical_candidates=PLANS.get(c, ()),
            discovery_only=("Tavily", "Parallel Search", "Firecrawl"),
            pit_status="PIT_UNPROVEN",
            notes="Candidate sources only; actual fixture/field coverage and publication timing must be measured.",
        )
        for c in TARGET_COMPETITIONS
    )
