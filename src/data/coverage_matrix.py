from __future__ import annotations

"""Actual coverage accounting for the fixed 15-competition research universe."""

COMPETITIONS = {
    "E0": "Premier League",
    "CH": "Championship",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "SP1": "La Liga",
    "F1": "Ligue 1",
    "N1": "Eredivisie",
    "UCL": "UEFA Champions League",
    "UEL": "UEFA Europa League",
    "J1": "J1",
    "J2": "J2",
    "J3": "J3",
    "DFBP": "DFB-Pokal",
    "FRIENDLY": "Club Friendlies",
    "EFL": "Carabao Cup / EFL Cup",
}

STATUSES = {"AVAILABLE", "PARTIAL", "UNAVAILABLE", "REAL_ZERO", "MISSING", "PARSE_ERROR", "COVERAGE_GAP"}


def make_row(competition: str, season: str, source: str, field: str, *, rows: int = 0,
             status: str = "UNAVAILABLE", pit_verified_rows: int = 0, note: str = "") -> dict:
    if competition not in COMPETITIONS:
        raise ValueError(f"unknown competition: {competition}")
    if status not in STATUSES:
        raise ValueError(f"unknown coverage status: {status}")
    return {
        "competition": competition,
        "competition_name": COMPETITIONS[competition],
        "season": season,
        "source": source,
        "field": field,
        "rows": int(rows),
        "pit_verified_rows": int(pit_verified_rows),
        "status": status,
        "note": note,
    }


def summarize(rows: list[dict]) -> dict:
    """Summarize actual observations; never infer coverage from API existence."""
    total = sum(int(r.get("rows", 0)) for r in rows)
    pit = sum(int(r.get("pit_verified_rows", 0)) for r in rows)
    return {
        "rows": total,
        "pit_verified_rows": pit,
        "records": len(rows),
        "available_records": sum(r.get("status") == "AVAILABLE" for r in rows),
        "partial_records": sum(r.get("status") == "PARTIAL" for r in rows),
        "unavailable_records": sum(r.get("status") == "UNAVAILABLE" for r in rows),
    }
