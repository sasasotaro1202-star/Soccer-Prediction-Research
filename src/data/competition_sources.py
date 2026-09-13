from __future__ import annotations

from dataclasses import dataclass

TARGET_COMPETITIONS = (
    "EPL", "CHA", "BL1", "SA", "LL", "FL1", "UCL", "UEL",
    "J1", "J2", "J3", "DFBP", "CAR", "FRI",
)

# Supplemental competitions are deliberately outside the strict 14-competition
# completion gate. They are acquired into a separate auxiliary dataset so they
# can improve context without silently contaminating the production club model.
AUXILIARY_COMPETITIONS = (
    "WORLD_CUP", "WORLD_CUP_QUALI", "ASIAN_CUP", "EURO", "EURO_QUALI",
    "NATIONS_LEAGUE", "INTERNATIONAL_FRIENDLY",
    "EMPERORS_CUP", "INTERHIGH", "JFA_U20", "JFA_U18",
)

AUXILIARY_NAMES = {
    "WORLD_CUP": "FIFA World Cup",
    "WORLD_CUP_QUALI": "FIFA World Cup Qualifiers",
    "ASIAN_CUP": "AFC Asian Cup",
    "EURO": "UEFA European Championship (EURO)",
    "EURO_QUALI": "UEFA EURO Qualifiers",
    "NATIONS_LEAGUE": "UEFA Nations League",
    "INTERNATIONAL_FRIENDLY": "Senior International Friendlies",
    "EMPERORS_CUP": "Emperor's Cup JFA All Japan Football Championship",
    "INTERHIGH": "National High School Sports Festival (Interhigh) Football",
    "JFA_U20": "Japan U-20 national-team matches",
    "JFA_U18": "Japan U-18 national-team matches",
}


@dataclass(frozen=True)
class CompetitionSourcePlan:
    competition: str
    canonical_candidates: tuple[str, ...]
    discovery_only: tuple[str, ...]
    pit_status: str
    notes: str


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

AUXILIARY_PLANS = {
    "WORLD_CUP": ("openfootball/internationals", "FIFA", "worldcup.json"),
    "WORLD_CUP_QUALI": ("openfootball/internationals", "FIFA"),
    "ASIAN_CUP": ("openfootball/internationals", "AFC", "JFA"),
    "EURO": ("openfootball/internationals", "UEFA"),
    "EURO_QUALI": ("openfootball/internationals", "UEFA"),
    "NATIONS_LEAGUE": ("openfootball/internationals", "UEFA"),
    "INTERNATIONAL_FRIENDLY": ("openfootball/internationals", "JFA", "FIFA"),
    "EMPERORS_CUP": ("JFA",),
    "INTERHIGH": ("JFA", "全国高等学校体育連盟サッカー専門部"),
    "JFA_U20": ("JFA", "FIFA", "AFC"),
    "JFA_U18": ("JFA", "UEFA", "AFC"),
}


def source_plans() -> tuple[CompetitionSourcePlan, ...]:
    return tuple(
        CompetitionSourcePlan(
            competition=c,
            canonical_candidates=PLANS[c],
            discovery_only=("Tavily", "Parallel Search", "Firecrawl"),
            pit_status="PIT_UNPROVEN",
            notes="Candidate sources only; actual coverage and publication timing must be measured.",
        )
        for c in TARGET_COMPETITIONS
    )


def auxiliary_source_plans() -> tuple[CompetitionSourcePlan, ...]:
    return tuple(
        CompetitionSourcePlan(
            competition=c,
            canonical_candidates=AUXILIARY_PLANS[c],
            discovery_only=("Tavily", "Parallel Search", "Firecrawl"),
            pit_status="PIT_UNKNOWN",
            notes="Auxiliary context only. Never promoted to production training until point-in-time publication timing is proven.",
        )
        for c in AUXILIARY_COMPETITIONS
    )
