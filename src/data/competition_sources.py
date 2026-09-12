from __future__ import annotations

from dataclasses import dataclass

TARGET_COMPETITIONS = (
    "EPL", "CHA", "BL1", "SA", "LL", "FL1", "UCL", "UEL",
    "J1", "J2", "J3", "DFBP", "CAR", "FRI",
)


@dataclass(frozen=True)
class CompetitionSourcePlan:
    competition: str
    canonical_candidates: tuple[str, ...]
    discovery_only: tuple[str, ...]
    pit_status: str
    notes: str


# Candidate plan only. It is never a coverage claim. Actual row/field coverage
# must be established by adapter execution and written to audit artifacts.
PLANS = {
    "EPL": ("Football-Data.co.uk", "football-data.org", "Sportmonks", "openfootball"),
    "CHA": ("Football-Data.co.uk", "football-data.org", "Sportmonks", "openfootball"),
    "BL1": ("Football-Data.co.uk", "football-data.org", "Sportmonks"),
    "SA": ("Football-Data.co.uk", "football-data.org", "Sportmonks"),
    "LL": ("Football-Data.co.uk", "football-data.org", "Sportmonks"),
    "FL1": ("Football-Data.co.uk", "football-data.org", "Sportmonks"),
    "UCL": ("football-data.org", "Sportmonks", "TheStatsAPI", "openfootball"),
    "UEL": ("football-data.org", "Sportmonks", "openfootball"),
    "J1": ("Sportmonks", "TheStatsAPI", "J.League"),
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
