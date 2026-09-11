from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

COMPETITIONS = {
    "EPL": "Premier League",
    "CHA": "Championship",
    "BL1": "Bundesliga",
    "SA": "Serie A",
    "LL": "La Liga",
    "FL1": "Ligue 1",
    "ERE": "Eredivisie",
    "UCL": "UEFA Champions League",
    "UEL": "UEFA Europa League",
    "J1": "J1",
    "J2": "J2",
    "J3": "J3",
    "DFBP": "DFB-Pokal",
    "FRI": "Club Friendlies",
    "CAR": "Carabao Cup / EFL Cup",
}

@dataclass(frozen=True)
class PredictionContext:
    match_id: str
    competition: str
    season: str
    kickoff_utc: datetime
    prediction_cutoff_at_utc: datetime
    data_snapshot_id: str

@dataclass
class MatchRecord:
    match_id: str
    competition: str
    season: str
    kickoff_utc: datetime
    home_team: str
    away_team: str
    home_goals: float | None = None
    away_goals: float | None = None
    source_name: str = ""
    source_record_id: str = ""
    source_available_at_utc: datetime | None = None
    retrieved_at_utc: datetime | None = None
    raw: dict[str, Any] | None = None
