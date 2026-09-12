from __future__ import annotations

"""Deterministic fixture/field/source/PIT audit for the 14 target competitions.

The audit measures observed acquisition; it never turns provider claims into
coverage claims and never converts missing values to zero.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.competition_sources import source_plans
from src.data.football_data import load_available_history
from src.data.pit_policy import is_available_by_cutoff

TARGET_COMPETITIONS = (
    "EPL", "CHA", "BL1", "SA", "LL", "FL1", "UCL", "UEL",
    "J1", "J2", "J3", "DFBP", "CAR", "FRI",
)

COMPETITION_NAMES = {
    "EPL": "Premier League", "CHA": "Championship", "BL1": "Bundesliga",
    "SA": "Serie A", "LL": "La Liga", "FL1": "Ligue 1",
    "UCL": "UEFA Champions League", "UEL": "UEFA Europa League",
    "J1": "J1", "J2": "J2", "J3": "J3", "DFBP": "DFB-Pokal",
    "CAR": "Carabao Cup / EFL Cup", "FRI": "Club Friendlies",
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
    if field.startswith(("home_", "away_")) and field.endswith(("goals", "shots", "shots_on_target", "corners", "fouls", "yellow_cards", "red_cards")):
        try:
            if float(value) == 0:
                return "REAL_ZERO"
        except (TypeError, ValueError):
            pass
    return "AVAILABLE"


def _fixture_id(row: pd.Series) -> str:
    return f"{row.get('source_name', 'unknown')}:{row.get('source_record_id', row.get('match_id', 'unknown'))}"


def fixture_audit(history: pd.DataFrame) -> pd.DataFrame:
    columns = ["fixture_id", "competition", "season", "home_team", "away_team", "kickoff_utc", "source", "source_fixture_id", "fixture_status", "kickoff_precision"]
    if history.empty:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([{
        "fixture_id": _fixture_id(row), "competition": row.get("competition"), "season": row.get("season"),
        "home_team": row.get("home_team"), "away_team": row.get("away_team"), "kickoff_utc": row.get("kickoff_utc"),
        "source": row.get("source_name"), "source_fixture_id": row.get("source_record_id", row.get("match_id")),
        "fixture_status": "OBSERVED", "kickoff_precision": row.get("event_time_precision"),
    } for _, row in history.iterrows()])


def field_audit(history: pd.DataFrame, config: AuditConfig = AuditConfig()) -> pd.DataFrame:
    rows: list[dict] = []
    if history.empty:
        return pd.DataFrame()
    for _, row in history.iterrows():
        fixture_id = _fixture_id(row)
        kickoff = pd.to_datetime(row.get("kickoff_utc"), utc=True, errors="coerce")
        cutoff = kickoff - pd.Timedelta(minutes=config.prediction_cutoff_minutes) if pd.notna(kickoff) else pd.NaT
        source_available = pd.to_datetime(row.get("source_available_at_utc"), utc=True, errors="coerce")
        retrieved = pd.to_datetime(row.get("retrieved_at_utc"), utc=True, errors="coerce")
        for field in CANONICAL_FIELDS[6:]:
            value = row.get(field, pd.NA)
            value_status = _status_for_value(value, field)
            if pd.isna(cutoff):
                pit_status = "PIT_UNKNOWN"
                pit_reason = "Invalid kickoff/cutoff"
            elif pd.notna(source_available):
                pit_status = "PIT_SAFE" if source_available <= cutoff else "PIT_UNSAFE"
                pit_reason = "Explicit source availability timestamp"
            elif field in {"home_goals", "away_goals", "result"}:
                pit_status = "PIT_UNSAFE"
                pit_reason = "Own-match outcome is post-event information and cannot be a pre-match feature"
            else:
                pit_status = "PIT_UNKNOWN"
                pit_reason = "No source publication timestamp; retrieval time is not treated as publication time"
            rows.append({
                "fixture_id": fixture_id, "competition": row.get("competition"), "season": row.get("season"),
                "home_team": row.get("home_team"), "away_team": row.get("away_team"), "event_time_utc": kickoff,
                "prediction_cutoff_at_utc": cutoff, "field_name": field, "value": value, "value_status": value_status,
                "source": row.get("source_name"), "source_fixture_id": row.get("source_record_id", row.get("match_id")),
                "source_available_at_utc": source_available, "retrieved_at_utc": retrieved,
                "pit_status": pit_status, "pit_reason": pit_reason,
            })
    return pd.DataFrame(rows)


def pit_audit(field_rows: pd.DataFrame) -> pd.DataFrame:
    if field_rows.empty:
        return pd.DataFrame(columns=["competition", "season", "field_name", "pit_status", "count", "rate"])
    rows = []
    for (competition, season, field, status), g in field_rows.groupby(["competition", "season", "field_name", "pit_status"], dropna=False):
        rows.append({"competition": competition, "season": season, "field_name": field, "pit_status": status, "count": int(len(g))})
    out = pd.DataFrame(rows)
    totals = out.groupby(["competition", "season", "field_name"], as_index=False)["count"].sum().rename(columns={"count": "total"})
    out = out.merge(totals, on=["competition", "season", "field_name"], how="left")
    out["rate"] = out["count"] / out["total"]
    return out


def coverage_matrix(history: pd.DataFrame, field_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    observed_comps = set(history["competition"].dropna().astype(str)) if not history.empty and "competition" in history.columns else set()
    for comp in TARGET_COMPETITIONS:
        comp_history = history[history["competition"].astype(str) == comp] if not history.empty and "competition" in history.columns else pd.DataFrame()
        if comp not in observed_comps:
            rows.append({"competition": comp, "competition_name": COMPETITION_NAMES[comp], "source": "current_observed_adapter", "status": "UNAVAILABLE", "fixture_count": 0, "reason": "No observed rows from current adapter; this is not a claim that the competition has no data elsewhere."})
            continue
        total = len(comp_history)
        rows.append({"competition": comp, "competition_name": COMPETITION_NAMES[comp], "source": "current_observed_adapter", "status": "AVAILABLE", "fixture_count": int(total), "reason": "Observed rows from current adapter"})
        if not field_rows.empty:
            f = field_rows[field_rows["competition"].astype(str) == comp]
            for field, g in f.groupby("field_name", dropna=False):
                nonmissing = int((g["value_status"] != "MISSING").sum())
                rows.append({"competition": comp, "competition_name": COMPETITION_NAMES[comp], "source": "current_observed_adapter", "field": field, "status": "AVAILABLE" if nonmissing == len(g) else ("PARTIAL" if nonmissing else "UNAVAILABLE"), "fixture_count": int(len(g)), "matched_count": nonmissing, "coverage_rate": float(nonmissing / len(g)) if len(g) else 0.0, "reason": "Observed field-level coverage; missing values are not converted to zero"})
    return pd.DataFrame(rows)


def source_reconciliation(history: pd.DataFrame) -> pd.DataFrame:
    columns = ["canonical_key", "source_count", "sources", "duplicate_source_identity", "row_count"]
    if history.empty:
        return pd.DataFrame(columns=columns)
    x = history.copy()
    x["canonical_key"] = x["competition"].astype(str) + "|" + x["season"].astype(str) + "|" + x["kickoff_utc"].astype(str) + "|" + x["home_team"].astype(str) + "|" + x["away_team"].astype(str)
    return pd.DataFrame([{
        "canonical_key": key, "source_count": len(set(g["source_name"].astype(str))),
        "sources": "|".join(sorted(set(g["source_name"].astype(str)))),
        "duplicate_source_identity": bool(len(g) > len(set(g["source_name"].astype(str)))), "row_count": int(len(g)),
    } for key, g in x.groupby("canonical_key", sort=False)])


def run_audit(out_dir: str = "artifacts", config: AuditConfig = AuditConfig()) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    history, acquisition = load_available_history(start_year=config.start_year, end_year=config.end_year)
    fixtures = fixture_audit(history)
    fields = field_audit(history, config=config)
    coverage = coverage_matrix(history, fields)
    reconciliation = source_reconciliation(history)
    pit = pit_audit(fields)
    plans = pd.DataFrame([{
        "competition": p.competition, "canonical_candidates": "|".join(p.canonical_candidates),
        "discovery_only": "|".join(p.discovery_only), "pit_status": p.pit_status, "notes": p.notes,
    } for p in source_plans()])

    fixtures.to_csv(out / "fixture_audit.csv", index=False)
    fields.to_csv(out / "field_audit.csv", index=False)
    coverage.to_csv(out / "coverage_matrix.csv", index=False)
    acquisition.to_csv(out / "acquisition_coverage.csv", index=False)
    reconciliation.to_csv(out / "source_reconciliation.csv", index=False)
    pit.to_csv(out / "pit_audit.csv", index=False)
    plans.to_csv(out / "competition_source_plan.csv", index=False)

    pit_counts = fields["pit_status"].value_counts().to_dict() if not fields.empty else {}
    observed = sorted(set(history["competition"].astype(str))) if not history.empty else []
    summary = {
        "target_competitions": list(TARGET_COMPETITIONS), "target_competition_count": 14,
        "observed_competitions": observed, "observed_competition_count": len(observed),
        "unobserved_competitions": [c for c in TARGET_COMPETITIONS if c not in observed],
        "fixture_count": int(len(fixtures)), "field_observation_count": int(len(fields)),
        "pit_status_counts": {str(k): int(v) for k, v in pit_counts.items()},
        "no_missing_to_zero": True, "search_engines_are_not_dataset_sources": True,
        "audit_scope": "fixture -> field -> source -> PIT -> reconciliation",
        "coverage_claim_policy": "observed-only; no provider-advertised coverage is promoted to AVAILABLE",
    }
    import json
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    import json
    print(json.dumps(run_audit(), indent=2, ensure_ascii=False))
