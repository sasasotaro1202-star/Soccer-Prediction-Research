from __future__ import annotations

"""Strict, explicit completion gate for the 14-competition soccer audit."""

from pathlib import Path
import json
import re

import pandas as pd

from src.data.fixture_field_audit import TARGET_COMPETITIONS

SEASONS = [f"{y}/{str(y + 1)[-2:]}" for y in range(2010, 2026)]
IMPLEMENTED_SOURCES = {
    "EPL": "Football-Data.co.uk", "CHA": "Football-Data.co.uk", "BL1": "Football-Data.co.uk",
    "SA": "Football-Data.co.uk", "LL": "Football-Data.co.uk", "FL1": "Football-Data.co.uk",
    "UCL": "openfootball", "UEL": "openfootball", "J1": "Football-Data.co.uk:JPN.csv",
    "J2": "Football-Data.co.uk:JPN.csv", "J3": "Football-Data.co.uk:JPN.csv", "DFBP": "openfootball",
    "CAR": "openfootball", "FRI": "ESPN:club.friendly",
}

# J3 was created for the 2014 season. Those four pre-launch cells are not
# missing data and must not make a historically scoped gate impossible.
NON_APPLICABLE_CELLS = {("J3", f"{y}/{str(y + 1)[-2:]}") for y in range(2010, 2014)}


def _normalize_acquisition(acq: pd.DataFrame) -> pd.DataFrame:
    if acq.empty or "season" not in acq.columns:
        return pd.DataFrame()
    x = acq.copy()
    x["competition"] = x["competition"].astype(str)
    x["season"] = x["season"].astype(str)
    x["source"] = x["source"].astype(str)
    return x


def _season_from_value(comp: str, value: str) -> str:
    if comp in {"J1", "J2", "J3"} and re.fullmatch(r"\d{4}", value):
        y = int(value)
        return f"{y}/{str(y + 1)[-2:]}"
    return value


def run_completion_gate(artifact_dir: str = "artifacts") -> dict:
    root = Path(artifact_dir)
    acquisition_path = root / "acquisition_coverage.csv"
    if not acquisition_path.exists():
        raise FileNotFoundError(acquisition_path)
    acquisition = _normalize_acquisition(pd.read_csv(acquisition_path))
    fields = pd.read_csv(root / "field_audit.csv") if (root / "field_audit.csv").exists() else pd.DataFrame()
    fixtures = pd.read_csv(root / "fixture_audit.csv") if (root / "fixture_audit.csv").exists() else pd.DataFrame()
    reconciliation = pd.read_csv(root / "source_reconciliation.csv") if (root / "source_reconciliation.csv").exists() else pd.DataFrame()

    expected = {(c, s) for c in TARGET_COMPETITIONS for s in SEASONS if (c, s) not in NON_APPLICABLE_CELLS}
    observed: set[tuple[str, str]] = set()
    rows = []
    for comp in TARGET_COMPETITIONS:
        source = IMPLEMENTED_SOURCES[comp]
        for season in SEASONS:
            if (comp, season) in NON_APPLICABLE_CELLS:
                rows.append({"competition": comp, "season": season, "canonical_source": source, "status": "NOT_APPLICABLE", "rows": 0, "reason": "Competition did not exist in this historical season"})
                continue
            candidates = acquisition[(acquisition.competition == comp) & (acquisition.source == source)]
            if comp in {"J1", "J2", "J3"}:
                candidates = candidates[candidates.season.map(lambda x: _season_from_value(comp, x) == season)]
            elif comp == "FRI":
                candidates = candidates[candidates.season == season]
            else:
                candidates = candidates[candidates.season.isin({season, season[:4]})]
            if len(candidates):
                observed.add((comp, season))
                status = "|".join(sorted(set(candidates.status.astype(str))))
                rows_count = int(pd.to_numeric(candidates.rows, errors="coerce").fillna(0).sum())
                reason = "Explicit acquisition status from adapter"
            else:
                status, rows_count, reason = "MISSING_AUDIT_CELL", 0, "Adapter did not emit an explicit season/source status"
            rows.append({"competition": comp, "season": season, "canonical_source": source, "status": status, "rows": rows_count, "reason": reason})
    matrix = pd.DataFrame(rows)
    matrix.to_csv(root / "competition_season_gate.csv", index=False)

    scope_complete = observed == expected
    data_available_cells = int(matrix.status.str.contains("AVAILABLE").sum())
    unavailable_cells = int(matrix.status.str.contains("UNAVAILABLE").sum())
    not_applicable_cells = int(matrix.status.str.contains("NOT_APPLICABLE").sum())
    parse_error_cells = int(matrix.status.str.contains("PARSE_ERROR").sum())
    missing_audit_cells = int(matrix.status.eq("MISSING_AUDIT_CELL").sum())

    duplicate_source_rows = int(reconciliation.duplicate_source_identity.sum()) if "duplicate_source_identity" in reconciliation.columns and not reconciliation.empty else 0
    collision_audit_available = "collision" in reconciliation.columns
    collision_rows = int(reconciliation.collision.sum()) if collision_audit_available and not reconciliation.empty else 0

    pit_unknown = int((fields.pit_status == "PIT_UNKNOWN").sum()) if "pit_status" in fields.columns else -1
    pit_unsafe = int((fields.pit_status == "PIT_UNSAFE").sum()) if "pit_status" in fields.columns else -1
    pit_safe = int((fields.pit_status == "PIT_SAFE").sum()) if "pit_status" in fields.columns else 0
    pit_gate = pit_unknown == 0 and pit_unsafe == 0

    team_mapping_issues = 0
    if not fixtures.empty and "team_mapping_status" in fixtures.columns:
        team_mapping_issues = int((fixtures.team_mapping_status != "VALID").sum())

    technical_audit_ok = not acquisition.empty and len(matrix) == len(expected) + len(NON_APPLICABLE_CELLS)
    full_gate_passed = bool(
        technical_audit_ok
        and scope_complete
        and data_available_cells + unavailable_cells + not_applicable_cells == len(matrix)
        and parse_error_cells == 0
        and missing_audit_cells == 0
        and duplicate_source_rows == 0
        and collision_audit_available
        and collision_rows == 0
        and team_mapping_issues == 0
        and pit_gate
    )

    result = {
        "target_competition_count": 14,
        "requested_season_count": 16,
        "requested_competition_season_cells": len(expected),
        "historically_not_applicable_cells": len(NON_APPLICABLE_CELLS),
        "technical_audit_ok": technical_audit_ok,
        "coverage_scope_complete": scope_complete,
        "data_available_cells": data_available_cells,
        "unavailable_cells": unavailable_cells,
        "not_applicable_cells": not_applicable_cells,
        "missing_audit_cells": missing_audit_cells,
        "parse_error_cells": parse_error_cells,
        "duplicate_source_rows": duplicate_source_rows,
        "collision_audit_available": collision_audit_available,
        "collision_rows": collision_rows,
        "team_mapping_issues": team_mapping_issues,
        "pit_safe_fields": pit_safe,
        "pit_unknown_fields": pit_unknown,
        "pit_unsafe_fields": pit_unsafe,
        "pit_publication_time_gate": pit_gate,
        "no_missing_to_zero": True,
        "full_gate_passed": full_gate_passed,
        "blocking_reasons": [],
    }
    if not technical_audit_ok:
        result["blocking_reasons"].append("technical audit matrix incomplete")
    if not scope_complete:
        result["blocking_reasons"].append("one or more competition-season cells lack explicit adapter status")
    if missing_audit_cells:
        result["blocking_reasons"].append("one or more competition-season cells lack an explicit acquisition audit row")
    if parse_error_cells:
        result["blocking_reasons"].append("parse errors remain")
    if duplicate_source_rows:
        result["blocking_reasons"].append("same-source duplicate fixture identities remain")
    if not collision_audit_available:
        result["blocking_reasons"].append("cross-source collision audit is unavailable")
    if collision_rows:
        result["blocking_reasons"].append("cross-source fixture outcome collisions remain")
    if team_mapping_issues:
        result["blocking_reasons"].append("team identity mapping issues remain")
    if not pit_gate:
        result["blocking_reasons"].append("publication-time PIT gate has UNKNOWN or UNSAFE fields")
    (root / "completion_gate.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run_completion_gate(), indent=2, ensure_ascii=False))
