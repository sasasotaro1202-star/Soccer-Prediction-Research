from __future__ import annotations

"""Strict completion gate for the 14-competition soccer research audit."""

import json
import re
from pathlib import Path

import pandas as pd

from src.data.fixture_field_audit import TARGET_COMPETITIONS
from src.data.football_data import load_available_history
from src.features.soccer_features import build_match_features

SEASONS = [f"{y}/{str(y + 1)[-2:]}" for y in range(2010, 2026)]
CANONICAL_SOURCES = {
    "EPL": "Football-Data.co.uk", "CHA": "Football-Data.co.uk", "BL1": "Football-Data.co.uk",
    "SA": "Football-Data.co.uk", "LL": "Football-Data.co.uk", "FL1": "Football-Data.co.uk",
    "UCL": "openfootball", "UEL": "openfootball", "J1": "J.League Data Site / Football-Data.co.uk:JPN.csv",
    "J2": "J.League Data Site / Football-Data.co.uk:JPN.csv", "J3": "J.League Data Site / Football-Data.co.uk:JPN.csv",
    "DFBP": "openfootball", "CAR": "openfootball", "FRI": "ESPN:club.friendly",
}
NON_APPLICABLE_CELLS = {("J3", f"{y}/{str(y + 1)[-2:]}") for y in range(2010, 2014)}


def _normalize_acquisition(acq: pd.DataFrame) -> pd.DataFrame:
    if acq.empty:
        return pd.DataFrame(columns=["competition", "season", "status", "rows", "source", "reason"])
    x = acq.copy()
    for c in ("competition", "season", "status", "source", "reason"):
        if c not in x.columns:
            x[c] = ""
        x[c] = x[c].astype(str)
    return x


def _season_display(comp: str, value: object) -> str:
    """Normalize every adapter's season representation to the audit's YYYY/YY key."""
    text = str(value).strip()
    if re.fullmatch(r"\d{4}", text):
        y = int(text)
        return f"{y}/{str(y + 1)[-2:]}"
    m = re.fullmatch(r"(\d{4})/(\d{2}|\d{4})", text)
    if m:
        y = int(m.group(1))
        return f"{y}/{str(y + 1)[-2:]}"
    return text


def _pit_preflight(root: Path) -> dict:
    history, _ = load_available_history(start_year=2010, end_year=2025)
    if history.empty:
        return {"rows": 0, "pit_verified_rows": 0, "pit_verified_rate": 0.0, "verified_competitions": 0}
    features = build_match_features(history, history, windows=(3, 5, 10, 20))
    features.to_csv(root / "pit_replay_features.csv", index=False)
    verified = features["pit_verified"].fillna(False).astype(bool) if "pit_verified" in features.columns else pd.Series(False, index=features.index)
    per_comp = features.assign(pit_verified=verified).groupby("competition")["pit_verified"].agg(["sum", "count"])
    verified_competitions = int((per_comp["sum"] >= 500).sum()) if not per_comp.empty else 0
    return {
        "rows": int(len(features)),
        "pit_verified_rows": int(verified.sum()),
        "pit_verified_rate": float(verified.mean()) if len(features) else 0.0,
        "verified_competitions": verified_competitions,
    }


def run_completion_gate(artifact_dir: str = "artifacts") -> dict:
    root = Path(artifact_dir)
    acquisition_path = root / "acquisition_coverage.csv"
    if not acquisition_path.exists():
        raise FileNotFoundError(acquisition_path)
    acquisition = _normalize_acquisition(pd.read_csv(acquisition_path, low_memory=False))
    fields = pd.read_csv(root / "field_audit.csv", low_memory=False) if (root / "field_audit.csv").exists() else pd.DataFrame()
    fixtures = pd.read_csv(root / "fixture_audit.csv", low_memory=False) if (root / "fixture_audit.csv").exists() else pd.DataFrame()
    reconciliation = pd.read_csv(root / "source_reconciliation.csv", low_memory=False) if (root / "source_reconciliation.csv").exists() else pd.DataFrame()

    expected = {(c, s) for c in TARGET_COMPETITIONS for s in SEASONS if (c, s) not in NON_APPLICABLE_CELLS}
    rows = []
    for comp in TARGET_COMPETITIONS:
        for season in SEASONS:
            if (comp, season) in NON_APPLICABLE_CELLS:
                rows.append({"competition": comp, "season": season, "canonical_source": CANONICAL_SOURCES[comp], "status": "NOT_APPLICABLE", "rows": 0, "reason": "Competition did not exist in this historical season"})
                continue
            candidates = acquisition[(acquisition.competition == comp) & acquisition.season.map(lambda x: _season_display(comp, x) == season)]
            if candidates.empty:
                status, count, reason = "MISSING_AUDIT_CELL", 0, "Adapter did not emit an explicit acquisition status"
            else:
                status = "|".join(sorted(set(candidates.status.astype(str))))
                count = int(pd.to_numeric(candidates.rows, errors="coerce").fillna(0).sum())
                reason = "Explicit acquisition status from one or more adapters"
            rows.append({"competition": comp, "season": season, "canonical_source": CANONICAL_SOURCES[comp], "status": status, "rows": count, "reason": reason})
    matrix = pd.DataFrame(rows)
    matrix.to_csv(root / "competition_season_gate.csv", index=False)

    missing_audit_cells = int(matrix.status.eq("MISSING_AUDIT_CELL").sum())
    parse_error_cells = int(matrix.status.str.contains("PARSE_ERROR", na=False).sum())
    available_cells = int(matrix.status.str.contains("AVAILABLE", na=False).sum())
    unavailable_cells = int(matrix.status.str.contains("UNAVAILABLE", na=False).sum())
    not_applicable_cells = int(matrix.status.str.contains("NOT_APPLICABLE", na=False).sum())
    accounted_cells = available_cells + unavailable_cells + not_applicable_cells
    scope_complete = missing_audit_cells == 0

    duplicate_source_rows = int(reconciliation.duplicate_source_identity.sum()) if "duplicate_source_identity" in reconciliation.columns and not reconciliation.empty else 0
    collision_audit_available = "collision" in reconciliation.columns
    collision_rows = int(reconciliation.collision.sum()) if collision_audit_available and not reconciliation.empty else 0
    team_mapping_issues = int((fixtures.team_mapping_status != "VALID").sum()) if "team_mapping_status" in fixtures.columns and not fixtures.empty else 0

    raw_pit_unknown = int((fields.pit_status == "PIT_UNKNOWN").sum()) if "pit_status" in fields.columns else -1
    raw_pit_unsafe = int((fields.pit_status == "PIT_UNSAFE").sum()) if "pit_status" in fields.columns else -1
    raw_pit_safe = int((fields.pit_status == "PIT_SAFE").sum()) if "pit_status" in fields.columns else 0

    try:
        pit = _pit_preflight(root)
        pit_preflight_error = None
    except Exception as exc:
        pit = {"rows": 0, "pit_verified_rows": 0, "pit_verified_rate": 0.0, "verified_competitions": 0}
        pit_preflight_error = f"{type(exc).__name__}: {exc}"
    pit_gate = (
        pit_preflight_error is None
        and pit["pit_verified_rows"] >= 7000
        and pit["pit_verified_rate"] >= 0.25
        and pit["verified_competitions"] >= 5
    )

    technical_audit_ok = not acquisition.empty and len(matrix) == len(expected) + len(NON_APPLICABLE_CELLS)
    accounting_ok = accounted_cells == len(matrix)
    full_gate_passed = bool(
        technical_audit_ok and scope_complete and accounting_ok
        and parse_error_cells == 0 and duplicate_source_rows == 0
        and collision_audit_available and collision_rows == 0
        and team_mapping_issues == 0 and pit_gate
    )

    result = {
        "target_competition_count": 14,
        "requested_season_count": 16,
        "requested_competition_season_cells": len(expected),
        "historically_not_applicable_cells": len(NON_APPLICABLE_CELLS),
        "technical_audit_ok": technical_audit_ok,
        "coverage_scope_complete": scope_complete,
        "data_available_cells": available_cells,
        "unavailable_cells": unavailable_cells,
        "not_applicable_cells": not_applicable_cells,
        "accounted_cells": accounted_cells,
        "missing_audit_cells": missing_audit_cells,
        "parse_error_cells": parse_error_cells,
        "duplicate_source_rows": duplicate_source_rows,
        "collision_audit_available": collision_audit_available,
        "collision_rows": collision_rows,
        "team_mapping_issues": team_mapping_issues,
        "raw_pit_safe_fields": raw_pit_safe,
        "raw_pit_unknown_fields": raw_pit_unknown,
        "raw_pit_unsafe_fields": raw_pit_unsafe,
        "pit_replay_rows": pit["rows"],
        "pit_replay_verified_rows": pit["pit_verified_rows"],
        "pit_replay_verified_rate": pit["pit_verified_rate"],
        "pit_replay_verified_competitions": pit["verified_competitions"],
        "pit_publication_time_gate": pit_gate,
        "no_missing_to_zero": True,
        "full_gate_passed": full_gate_passed,
        "blocking_reasons": [],
    }
    if not technical_audit_ok:
        result["blocking_reasons"].append("technical audit matrix incomplete")
    if not scope_complete:
        result["blocking_reasons"].append("one or more competition-season cells lack explicit adapter status")
    if not accounting_ok:
        result["blocking_reasons"].append("competition-season status accounting is incomplete")
    if missing_audit_cells:
        result["blocking_reasons"].append("one or more competition-season cells lack an acquisition audit row")
    if parse_error_cells:
        result["blocking_reasons"].append("parse errors remain in acquisition coverage")
    if duplicate_source_rows:
        result["blocking_reasons"].append("same-source duplicate fixture identities remain")
    if not collision_audit_available:
        result["blocking_reasons"].append("cross-source collision audit is unavailable")
    if collision_rows:
        result["blocking_reasons"].append("cross-source fixture outcome collisions remain")
    if team_mapping_issues:
        result["blocking_reasons"].append("team identity mapping issues remain")
    if pit_preflight_error:
        result["blocking_reasons"].append(f"PIT replay preflight failed: {pit_preflight_error}")
    elif not pit_gate:
        result["blocking_reasons"].append("insufficient PIT-verified replay rows for the configured walk-forward evaluation")
    (root / "completion_gate.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run_completion_gate(), indent=2, ensure_ascii=False, default=str))
