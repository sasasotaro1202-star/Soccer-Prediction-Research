"""Broad soccer research competition catalog with conservative production eligibility.

The catalog is intentionally broader than currently verified PIT adapters. A
competition being listed here means it is a research target, not that it is safe
for production. Production eligibility must be earned through auditable PIT
source evidence, coverage/QC, feature readiness, chronological OOS validation,
and downstream adoption gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CompetitionSpec:
    code: str
    name: str
    region: str
    tier: str
    competition_type: str
    parent_code: str | None = None
    research_enabled: bool = True
    production_eligible: bool = False
    pit_status: str = "UNVERIFIED"
    data_quality_status: str = "UNVERIFIED"


# Broad research scope. Every new target remains production-ineligible until
# the evidence/adoption gates prove that it is safe for real-world prediction.
COMPETITION_CATALOG: tuple[CompetitionSpec, ...] = (
    # England
    CompetitionSpec("EPL", "Premier League", "England", "1", "league"),
    CompetitionSpec("CHA", "Championship", "England", "2", "league"),
    CompetitionSpec("ENG1", "League One", "England", "3", "league"),
    CompetitionSpec("ENG2", "League Two", "England", "4", "league"),
    CompetitionSpec("FAC", "FA Cup", "England", "cup", "cup"),
    CompetitionSpec("EFL", "EFL Cup / Carabao Cup", "England", "cup", "cup"),
    # Germany
    CompetitionSpec("BL1", "Bundesliga", "Germany", "1", "league"),
    CompetitionSpec("BL2", "2. Bundesliga", "Germany", "2", "league"),
    CompetitionSpec("DFBP", "DFB-Pokal", "Germany", "cup", "cup"),
    # Spain
    CompetitionSpec("LL", "La Liga", "Spain", "1", "league"),
    CompetitionSpec("LL2", "Segunda División", "Spain", "2", "league"),
    CompetitionSpec("CDR", "Copa del Rey", "Spain", "cup", "cup"),
    # Italy
    CompetitionSpec("SA", "Serie A", "Italy", "1", "league"),
    CompetitionSpec("SB", "Serie B", "Italy", "2", "league"),
    CompetitionSpec("COPPA", "Coppa Italia", "Italy", "cup", "cup"),
    # France
    CompetitionSpec("FL1", "Ligue 1", "France", "1", "league"),
    CompetitionSpec("FL2", "Ligue 2", "France", "2", "league"),
    CompetitionSpec("CDF", "Coupe de France", "France", "cup", "cup"),
    # Other European leagues/cups
    CompetitionSpec("ERE", "Eredivisie", "Netherlands", "1", "league"),
    CompetitionSpec("TACA", "KNVB Cup", "Netherlands", "cup", "cup"),
    CompetitionSpec("PPL", "Primeira Liga", "Portugal", "1", "league"),
    CompetitionSpec("TACA_POR", "Taça de Portugal", "Portugal", "cup", "cup"),
    CompetitionSpec("BEL", "Belgian Pro League", "Belgium", "1", "league"),
    CompetitionSpec("BELCUP", "Belgian Cup", "Belgium", "cup", "cup"),
    CompetitionSpec("SPL", "Scottish Premiership", "Scotland", "1", "league"),
    CompetitionSpec("SCOTCUP", "Scottish Cup", "Scotland", "cup", "cup"),
    CompetitionSpec("TUR", "Süper Lig", "Turkey", "1", "league"),
    CompetitionSpec("TURCUP", "Turkish Cup", "Turkey", "cup", "cup"),
    CompetitionSpec("GRE", "Super League Greece", "Greece", "1", "league"),
    CompetitionSpec("GRECUP", "Greek Cup", "Greece", "cup", "cup"),
    CompetitionSpec("AUT", "Austrian Bundesliga", "Austria", "1", "league"),
    CompetitionSpec("SUI", "Swiss Super League", "Switzerland", "1", "league"),
    # Japan
    CompetitionSpec("J1", "J1 League", "Japan", "1", "league"),
    CompetitionSpec("J2", "J2 League", "Japan", "2", "league"),
    CompetitionSpec("J3", "J3 League", "Japan", "3", "league"),
    CompetitionSpec("JLC", "J.League Cup", "Japan", "cup", "cup"),
    CompetitionSpec("EMP_CUP", "Emperor's Cup", "Japan", "cup", "cup"),
    # Korea / Americas / Saudi Arabia
    CompetitionSpec("KOR", "K League 1", "South Korea", "1", "league"),
    CompetitionSpec("KORCUP", "Korean FA Cup", "South Korea", "cup", "cup"),
    CompetitionSpec("MLS", "Major League Soccer", "United States", "1", "league"),
    CompetitionSpec("USOC", "U.S. Open Cup", "United States", "cup", "cup"),
    CompetitionSpec("LIGA_MX", "Liga MX", "Mexico", "1", "league"),
    CompetitionSpec("MEXCUP", "Copa MX", "Mexico", "cup", "cup"),
    CompetitionSpec("BRA", "Brazilian Série A", "Brazil", "1", "league"),
    CompetitionSpec("BRA_CUP", "Copa do Brasil", "Brazil", "cup", "cup"),
    CompetitionSpec("ARG", "Argentine Primera División", "Argentina", "1", "league"),
    CompetitionSpec("ARG_CUP", "Copa Argentina", "Argentina", "cup", "cup"),
    CompetitionSpec("SPL_SA", "Saudi Pro League", "Saudi Arabia", "1", "league"),
    # UEFA club competitions
    CompetitionSpec("UCL", "UEFA Champions League", "Europe", "continental", "europe"),
    CompetitionSpec("UEL", "UEFA Europa League", "Europe", "continental", "europe"),
    CompetitionSpec("UECL", "UEFA Conference League", "Europe", "continental", "europe"),

    # ------------------------------------------------------------------
    # International senior football: men and women.
    # ------------------------------------------------------------------
    CompetitionSpec("WC_M", "FIFA World Cup Men", "Global", "senior", "international"),
    CompetitionSpec("WC_W", "FIFA Women's World Cup", "Global", "senior", "international"),
    CompetitionSpec("CONT_M", "Senior Men's Continental Championships", "Global", "senior", "international"),
    CompetitionSpec("CONT_W", "Senior Women's Continental Championships", "Global", "senior", "international"),
    CompetitionSpec("INTL_M", "Senior Men's International Friendlies", "Global", "senior", "friendly"),
    CompetitionSpec("INTL_W", "Senior Women's International Friendlies", "Global", "senior", "friendly"),

    # Asian football, including the Asian Games men's and women's tournaments.
    CompetitionSpec("AG_M", "Asian Games Men's Football", "Asia", "senior", "multi_sport"),
    CompetitionSpec("AG_W", "Asian Games Women's Football", "Asia", "senior", "multi_sport"),
    CompetitionSpec("AFC_M", "AFC Men's Competitions", "Asia", "senior", "international"),
    CompetitionSpec("AFC_W", "AFC Women's Competitions", "Asia", "senior", "international"),
    CompetitionSpec("EAFF_M", "EAFF Men's Competitions", "East Asia", "senior", "international"),
    CompetitionSpec("EAFF_W", "EAFF Women's Competitions", "East Asia", "senior", "international"),

    # Youth national teams: explicitly separate age groups and gender.
    CompetitionSpec("U23_M", "Men's U-23 International Football", "Global", "U23", "youth"),
    CompetitionSpec("U23_W", "Women's U-23 International Football", "Global", "U23", "youth"),
    CompetitionSpec("U20_M", "Men's U-20 International Football", "Global", "U20", "youth"),
    CompetitionSpec("U20_W", "Women's U-20 International Football", "Global", "U20", "youth"),
    CompetitionSpec("U18_M", "Men's U-18 International Football", "Global", "U18", "youth"),
    CompetitionSpec("U18_W", "Women's U-18 International Football", "Global", "U18", "youth"),
    CompetitionSpec("U17_M", "Men's U-17 International Football", "Global", "U17", "youth"),
    CompetitionSpec("U17_W", "Women's U-17 International Football", "Global", "U17", "youth"),
    CompetitionSpec("U16_M", "Men's U-16 International Football", "Global", "U16", "youth"),
    CompetitionSpec("U16_W", "Women's U-16 International Football", "Global", "U16", "youth"),

    # Japan youth / school football. These are research targets only until
    # auditable historical snapshots and publication-time timestamps exist.
    CompetitionSpec("JFA_U23_M", "Japan U-23 Men's Football", "Japan", "U23", "youth"),
    CompetitionSpec("JFA_U20_M", "Japan U-20 Men's Football", "Japan", "U20", "youth"),
    CompetitionSpec("JFA_U18_M", "Japan U-18 Men's Football", "Japan", "U18", "youth"),
    CompetitionSpec("JFA_U18_W", "Japan U-18 Women's Football", "Japan", "U18", "youth"),
    CompetitionSpec("JFA_U17_M", "Japan U-17 Men's Football", "Japan", "U17", "youth"),
    CompetitionSpec("JFA_U17_W", "Japan U-17 Women's Football", "Japan", "U17", "youth"),
    CompetitionSpec("INTERHIGH_M", "Japan Inter-High School Men's Football", "Japan", "school", "school"),
    CompetitionSpec("INTERHIGH_W", "Japan Inter-High School Women's Football", "Japan", "school", "school"),
    CompetitionSpec("JFA_U18_CLUB_M", "Japan U-18 Club / Academy Football", "Japan", "U18", "youth"),
    CompetitionSpec("JFA_U18_CLUB_W", "Japan U-18 Women's Club / Academy Football", "Japan", "U18", "youth"),
    CompetitionSpec("JFA_YOUTH_M", "Japan Other Youth / Academy Football", "Japan", "youth", "youth"),
    CompetitionSpec("JFA_YOUTH_W", "Japan Other Women's Youth / Academy Football", "Japan", "youth", "youth"),
    CompetitionSpec("JFA_SCHOOL_M", "Japan Other School Football", "Japan", "school", "school"),
    CompetitionSpec("JFA_SCHOOL_W", "Japan Other Women's School Football", "Japan", "school", "school"),
    CompetitionSpec("JFA_CUP_YOUTH", "Japan Youth / School Cup Football", "Japan", "youth", "cup"),

    # Other major Japanese school/youth competitions are deliberately grouped
    # as research targets so future source adapters can expand coverage without
    # silently granting production eligibility.
    CompetitionSpec("CLUB_YOUTH_GLOBAL_M", "Global Men's Youth Club Competitions", "Global", "youth", "youth"),
    CompetitionSpec("CLUB_YOUTH_GLOBAL_W", "Global Women's Youth Club Competitions", "Global", "youth", "youth"),

    # Non-competitive fixtures are tracked separately and never auto-promoted.
    CompetitionSpec("FRIENDLY", "International / Club Friendly", "Global", "special", "friendly"),
)


def competition_catalog() -> list[dict[str, object]]:
    """Return a stable, serialization-friendly catalog for audits and reports."""
    return [asdict(spec) for spec in COMPETITION_CATALOG]


def production_candidates() -> list[CompetitionSpec]:
    """Return only entries explicitly marked production-eligible."""
    return [spec for spec in COMPETITION_CATALOG if spec.production_eligible]


def research_targets() -> list[CompetitionSpec]:
    """Return all enabled research targets, including currently unverified ones."""
    return [spec for spec in COMPETITION_CATALOG if spec.research_enabled]


# Active scope is intentionally narrower than the broad research catalog.
# Parked competitions remain catalogued for later expansion, but acquisition,
# OOS research and production work must not spend resources on them.
ACTIVE_SCOPE = frozenset({
    "EPL", "AG_M", "AG_W", "ERE", "LL", "SA", "BL1",
    "J1", "J2", "J3", "FL1", "UCL", "UEL", "U23_M", "U18_M",
})

def active_competitions() -> tuple[CompetitionSpec, ...]:
    return tuple(spec for spec in COMPETITION_CATALOG if spec.code in ACTIVE_SCOPE)
