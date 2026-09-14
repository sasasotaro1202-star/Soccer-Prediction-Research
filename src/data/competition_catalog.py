"""Research competition catalog with conservative production eligibility.

The catalog is intentionally broader than the currently verified PIT adapters. A
competition being listed here means it is a research target, not that it is safe
for production. Production eligibility must be earned through an auditable PIT
source, coverage/QC, feature readiness, and downstream validation gates.
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


# Broad research scope. Keep production_eligible=False until the existing
# adoption gate proves PIT-safe, sufficiently covered, reproducible data.
COMPETITION_CATALOG: tuple[CompetitionSpec, ...] = (
    # England
    CompetitionSpec("EPL", "Premier League", "England", "1", "league"),
    CompetitionSpec("CHA", "Championship", "England", "2", "league"),
    CompetitionSpec("ENG1", "League One", "England", "3", "league"),
    CompetitionSpec("ENG2", "League Two", "England", "4", "league"),
    CompetitionSpec("FAC", "FA Cup", "England", "cup", research_enabled=False),
    CompetitionSpec("EFL", "EFL Cup / Carabao Cup", "England", "cup", research_enabled=False),
    # Germany
    CompetitionSpec("BL1", "Bundesliga", "Germany", "1", "league"),
    CompetitionSpec("BL2", "2. Bundesliga", "Germany", "2", "league"),
    CompetitionSpec("DFBP", "DFB-Pokal", "Germany", "cup"),
    # Spain
    CompetitionSpec("LL", "La Liga", "Spain", "1", "league"),
    CompetitionSpec("LL2", "Segunda División", "Spain", "2", "league"),
    CompetitionSpec("CDR", "Copa del Rey", "Spain", "cup", research_enabled=False),
    # Italy
    CompetitionSpec("SA", "Serie A", "Italy", "1", "league"),
    CompetitionSpec("SB", "Serie B", "Italy", "2", "league"),
    CompetitionSpec("COPPA", "Coppa Italia", "Italy", "cup", research_enabled=False),
    # France
    CompetitionSpec("FL1", "Ligue 1", "France", "1", "league"),
    CompetitionSpec("FL2", "Ligue 2", "France", "2", "league"),
    CompetitionSpec("CDF", "Coupe de France", "France", "cup", research_enabled=False),
    # Netherlands / Portugal / Belgium / Scotland / Turkey / Greece
    CompetitionSpec("ERE", "Eredivisie", "Netherlands", "1", "league"),
    CompetitionSpec("TACA", "KNVB Cup", "Netherlands", "cup", research_enabled=False),
    CompetitionSpec("PPL", "Primeira Liga", "Portugal", "1", "league"),
    CompetitionSpec("TACA_POR", "Taça de Portugal", "Portugal", "cup", research_enabled=False),
    CompetitionSpec("BEL", "Belgian Pro League", "Belgium", "1", "league"),
    CompetitionSpec("BELCUP", "Belgian Cup", "Belgium", "cup", research_enabled=False),
    CompetitionSpec("SPL", "Scottish Premiership", "Scotland", "1", "league"),
    CompetitionSpec("SCOTCUP", "Scottish Cup", "Scotland", "cup", research_enabled=False),
    CompetitionSpec("TUR", "Süper Lig", "Turkey", "1", "league"),
    CompetitionSpec("TURCUP", "Turkish Cup", "Turkey", "cup", research_enabled=False),
    CompetitionSpec("GRE", "Super League Greece", "Greece", "1", "league"),
    CompetitionSpec("GRECUP", "Greek Cup", "Greece", "cup", research_enabled=False),
    # Central Europe
    CompetitionSpec("AUT", "Austrian Bundesliga", "Austria", "1", "league"),
    CompetitionSpec("SUI", "Swiss Super League", "Switzerland", "1", "league"),
    # Japan / Asia
    CompetitionSpec("J1", "J1 League", "Japan", "1", "league"),
    CompetitionSpec("J2", "J2 League", "Japan", "2", "league"),
    CompetitionSpec("J3", "J3 League", "Japan", "3", "league"),
    CompetitionSpec("JLC", "J.League Cup", "Japan", "cup", research_enabled=False),
    CompetitionSpec("EMP_CUP", "Emperor's Cup", "Japan", "cup", research_enabled=False),
    CompetitionSpec("KOR", "K League 1", "South Korea", "1", "league"),
    CompetitionSpec("KORCUP", "Korean FA Cup", "South Korea", "cup", research_enabled=False),
    # Americas
    CompetitionSpec("MLS", "Major League Soccer", "United States", "1", "league"),
    CompetitionSpec("USOC", "U.S. Open Cup", "United States", "cup", research_enabled=False),
    CompetitionSpec("LIGA_MX", "Liga MX", "Mexico", "1", "league"),
    CompetitionSpec("MEXCUP", "Copa MX", "Mexico", "cup", research_enabled=False),
    CompetitionSpec("BRA", "Brazilian Série A", "Brazil", "1", "league"),
    CompetitionSpec("BRA_CUP", "Copa do Brasil", "Brazil", "cup", research_enabled=False),
    CompetitionSpec("ARG", "Argentine Primera División", "Argentina", "1", "league"),
    CompetitionSpec("ARG_CUP", "Copa Argentina", "Argentina", "cup", research_enabled=False),
    # Saudi Arabia / other high-interest competitions
    CompetitionSpec("SPL_SA", "Saudi Pro League", "Saudi Arabia", "1", "league"),
    # UEFA
    CompetitionSpec("UCL", "UEFA Champions League", "Europe", "continental", "europe"),
    CompetitionSpec("UEL", "UEFA Europa League", "Europe", "continental", "europe"),
    CompetitionSpec("UECL", "UEFA Conference League", "Europe", "continental", "europe"),
    # Non-competitive fixtures are tracked separately and never receive an
    # automatic production pass merely because they have data.
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
