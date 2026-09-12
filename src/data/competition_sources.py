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


# Candidate/adoption plan only. Actual coverage is established by execution.
PLANS = {
    "EPL": ("Football-Data.co.uk", "football-data.org", "Sportmonks", "openfootball"),
    "CHA": ("Football-Data.co.uk", "football-data.org", "Sportmonks", "openfootball"),
    "BL1": ("Football-Data.co.uk", "football-data.org", "Sportmonks", "openfootball"),
    "SA": ("Football-Data.co.uk", "football-data.org", "Sportmonks", "openfootball"),
    "LL": ("Football-Data.co.uk", "football-data.org", "Sportmonks", "openfootball"),
    "FL1": ("Football-Data.co.uk", "football-data.org", "Sportmonks", "openfootball"),
    "UCL": ("openfootball", "football-data.org", "Sportmonks"),
    "UEL": ("openfootball", "football-data.org", "Sportmonks"),
    "J1": ("J.League", "Sportmonks", "TheStatsAPI"),
    "J2": ("J.League", "Sportmonks", "TheStatsAPI"),
    "J3": ("J.League", "Sportmonks", "TheStatsAPI"),
    "DFBP": ("openfootball", "Sportmonks"),
    "CAR": ("openfootball", "Sportmonks"),
    "FRI": ("ESPN", "Sportmonks"),
}


def source_plans() -> tuple[CompetitionSourcePlan, ...]:
    return tuple(
        CompetitionSourcePlan(
            competition=c,
            canonical_candidates=PLANS[c],
            discovery_only=("Tavily", "Parallel Search", "Firecrawl"),
            pit_status="PIT_UNPROVEN",
            notes="Candidate sources only; actual fixture/field coverage and publication timing must be measured.",
        )
        for c in TARGET_COMPETITIONS
    )
