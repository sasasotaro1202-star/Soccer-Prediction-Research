from __future__ import annotations

"""Deterministic fixture/field/source/PIT audit for the 14 target competitions.

This module deliberately separates *declared scope* from *observed acquisition*.
A competition is never marked AVAILABLE merely because a provider is expected to
cover it. Missing values remain missing and are classified explicitly.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.football_data import load_available_history
from src.data.pit_policy import is_available_by_cutoff

TARGET_COMPETITIONS = (
    "EPL", "CHA", "BL1", "SA", "LL", "FL1", "UCL", "UEL",
    "J1", "J2", "J3", "DFBP", "CAR", "FRI",
)

COMPETITION_NAMES = {
    "EPL": "Premier League",
    "CHA": "Championship",
    "BL1": "Bundesliga",
    "SA": "Serie A",
    "LL": "La Liga",
    "FL1": "Ligue 1",
    "UCL": "UEFA Champions League",
    "UEL": "UEFA Europa League",
    "J1": "J1",
    "J2": "J2",
    "J3": "J3",
    "DFBP": "DFB-Pokal",
    "CAR": "Carabao Cup / EFL Cup",
    "FRI": "Club Friendlies",
}

CANONICAL_FIELDS = (
    "fixture_id", "competition", "season", "home_team", "away_team",
    "kickoff_utc", "result", "home_goals", "away_goals",
    "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    "home_corners", "away_corners", "home_fouls", "away_fouls",
    "home_yellow_cards", "away_yellow_cards", "home_red_cards", "away_red_cards",
)


@dataclass(frozen=True)
class AuditConfig:
    start_year: int = 2010
    end_year: int = 2025
    prediction_cutoff_minutes: int = 60


def _status_for_value(value, field: str) -> str:
    if pd.isna(value):
        return "MISSING"
    if field.startswith("home_") or field.startswith("away_"):
        if field.endswith(("goals", "shots", "shots_on_target", "corners", "fouls", "yellow_cards", "red_cards")) and float(value) == 0:
            return "REAL_ZERO"
    return "AVAILABLE"


def _fixture_id(row: pd.Series) -> str:
    source = str(row.get("source_name", "unknown"))
    source_id = str(row.get("source_record_id", "unknown"))
    return f"{source}:{source_id}"


def fixture_audit(history: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if history.empty:
        return pd.DataFrame(columns=["fixture_id", "competition", "season", "home_team", "away_team", "kickoff_utc", "source", "source_fixture_id"])
    for _, row in history.iterrows():
        rows.append({
            "fixture_id": _fixture_id(row),
            "competition": row.get("competition"),
            "season": row.get("season"),
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "kickoff_utc": row.get("kickoff_utc"),
            "source": row.get("source_name"),
            "source_fixture_id": row.get("source_record_id"),
            "fixture_status": "OBSERVED",
            "kickoff_precision": row.get("event_time_precision"),
        })
    return pd.DataFrame(rows)


def field_audit(history: pd.DataFrame, config: AuditConfig = AuditConfig()) -> pd.DataFrame:
    rows: list[dict] = []
    if history.empty:
        return pd.DataFrame()
    for _, row in history.iterrows():
        fixture_id = _fixture_id(row)
        kickoff = pd.to_datetime(row.get("kickoff_utc"), utc=True, errors="coerce")
        cutoff = kickoff - pd.Timedelta(minutes=config.prediction_cutoff_minutes) if pd.notna(kickoff) else pd.NaT
        source_available = pd.to_datetime(row.get("source_available_at_utc"), utc=True, errors="coerce")
        for field in CANONICAL_FIELDS[6:]:
            value = row.get(field, pd.NA)
            value_status = _status_for_value(value, field)
            if pd.isna(cutoff):
                pit_status = "PIT_UNKNOWN"
            elif pd.notna(source_available):
                pit_status = "PIT_SAFE" if source_available <= cutoff else "PIT_UNSAFE"
            elif field in {"home_goals", "away_goals", "result"}:
                # Historical result-derived features use the explicit deterministic
                # policy rather than treating retrieval time as publication time.
                pit_status = "PIT_SAFE" if is_available_by_cutoff(kickoff, cutoff) else "PIT_UNSAFE"
            else:
                pit_status = "PIT_UNKNOWN"
            rows.append({
                "fixture_id": fixture_id,
                "competition": row.get("competition"),
                "season": row.get("season"),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "event_time_utc": kickoff,
                "prediction_cutoff_at_utc": cutoff,
                "field_name": field,
                "value": value,
                "value_status": value_status,
                "source": row.get("source_name"),
                "source_fixture_id": row.get("source_record_id"),
                "source_available_at_utc": source_available,
                "retrieved_at_utc": pd.to_datetime(row.get("retrieved_at_utc"), utc=True, errors="coerce"),
                "pit_status": pit_status,
            })
    return pd.DataFrame(rows)


def coverage_matrix(history: pd.DataFrame, field_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    observed_comps = set(history["competition"].dropna().astype(str)) if not history.empty else set()
    for comp in TARGET_COMPETITIONS:
        comp_history = history[history["competition"].astype(str) == comp] if not history.empty and "competition" in history.columns else pd.DataFrame()
        if comp not in observed_comps:
            rows.append({
                "competition": comp,
                "competition_name": COMPETITION_NAMES[comp],
                "source": "Football-Data.co.uk",
                "status": "UNAVAILABLE",
                "fixture_count": 0,
                "reason": "No observed rows from current adapter; this is not a claim that the competition has no data elsewhere.",
            })
            continue
        total = len(comp_history)
        rows.append({
            "competition": comp,
            "competition_name": COMPETITION_NAMES[comp],
            "source": "Football-Data.co.uk",
            "status": "AVAILABLE" if total else "UNAVAILABLE",
            "fixture_count": int(total),
            "reason": "Observed from current adapter" if total else "No observed rows",
        })
        if not field_rows.empty:
            f = field_rows[field_rows["competition"].astype(str) == comp]
            for field, g in f.groupby("field_name", dropna=False):
                nonmissing = int((g["value_status"] != "MISSING").sum())
                rows.append({
                    "competition": comp,
                    "competition_name": COMPETITION_NAMES[comp],
                    "source": "Football-Data.co.uk",
                    "field": field,
                    "status": "AVAILABLE" if nonmissing == len(g) else ("PARTIAL" if nonmissing else "UNAVAILABLE"),
                    "fixture_count": int(len(g)),
                    "matched_count": nonmissing,
                    "coverage_rate": float(nonmissing / len(g)) if len(g) else 0.0,
                    "reason": "Observed field-level coverage; missing values are not converted to zero",
                })
    return pd.DataFrame(rows)


def source_reconciliation(history: pd.DataFrame) -> pd.DataFrame:
    """Detect duplicate observations from the same canonical source identity.

    The current adapter has one source. The schema is intentionally ready for
    future multi-source reconciliation without counting the same URL/provider twice.
    """
    if history.empty:
        return pd.DataFrame(columns=["canonical_key", "source_count", "sources", "duplicate_source_identity"])
    x = history.copy()
    x["canonical_key"] = (
        x["competition"].astype(str) + "|" + x["season"].astype(str) + "|" +
        x["kickoff_utc"].astype(str) + "|" + x["home_team"].astype(str) + "|" + x["away_team"].astype(str)
    )
    rows = []
    for key, g in x.groupby("canonical_key", sort=False):
        sources = sorted(set(g["source_name"].astype(str)))
        rows.append({
            "canonical_key": key,
            "source_count": len(sources),
            "sources": "|".join(sources),
            "duplicate_source_identity": bool(len(g) > len(sources)),
            "row_count": int(len(g)),
        })
    return pd.DataFrame(rows)


def run_audit(out_dir: str = "artifacts", config: AuditConfig = AuditConfig()) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    history, acquisition = load_available_history(start_year=config.start_year, end_year=config.end_year)
    fixtures = fixture_audit(history)
    fields = field_audit(history, config=config)
    coverage = coverage_matrix(history, fields)
    reconciliation = source_reconciliation(history)

    fixtures.to_csv(out / "fixture_audit.csv", index=False)
    fields.to_csv(out / "field_audit.csv", index=False)
    coverage.to_csv(out / "coverage_matrix.csv", index=False)
    acquisition.to_csv(out / "acquisition_coverage.csv", index=False)
    reconciliation.to_csv(out / "source_reconciliation.csv", index=False)

    pit_counts = fields["pit_status"].value_counts().to_dict() if not fields.empty else {}
    observed = sorted(set(history["competition"].astype(str))) if not history.empty else []
    summary = {
        "target_competitions": list(TARGET_COMPETITIONS),
        "target_competition_count": len(TARGET_COMPETITIONS),
        "observed_competitions": observed,
        "observed_competition_count": len(observed),
        "unobserved_competitions": [c for c in TARGET_COMPETITIONS if c not in observed],
        "fixture_count": int(len(fixtures)),
        "field_observation_count": int(len(fields)),
        "pit_status_counts": {str(k): int(v) for k, v in pit_counts.items()},
        "no_missing_to_zero": True,
        "search_engines_are_not_dataset_sources": True,
        "audit_scope": "fixture -> field -> source -> PIT -> reconciliation",
    }
    import json
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    import json
    print(json.dumps(run_audit(), indent=2, ensure_ascii=False))
