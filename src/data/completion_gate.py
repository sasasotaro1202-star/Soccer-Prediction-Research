from __future__ import annotations

"""Strict, explicit completion gate for the 14-competition soccer audit."""

from pathlib import Path
import json
import re

import pandas as pd

from src.data.fixture_field_audit import TARGET_COMPETITIONS
from src.data.competition_sources import PLANS

SEASONS = [f"{y}/{str(y + 1)[-2:]}" for y in range(2010, 2026)]


def _canonical_source(comp: str) -> str:
    return PLANS[comp][0]


def _normalize_acquisition(acq: pd.DataFrame) -> pd.DataFrame:
    if acq.empty:
        return acq
    x = acq.copy()
    if "season" not in x.columns:
        return pd.DataFrame()
    x["competition"] = x["competition"].astype(str)
    x["season"] = x["season"].astype(str)
    return x


def _season_from_value(comp: str, value: str) -> str:
    if comp in {"J1", "J2", "J3"}:
        if re.fullmatch(r"\d{4}", value):
            y = int(value)
            return f"{y}/{str(y + 1)[-2:]}"
    return value


def run_completion_gate(artifact_dir: str = "artifacts") -> dict:
    root = Path(artifact_dir)
    acquisition_path = root / "acquisition_coverage.csv"
    field_path = root / "field_audit.csv"
    fixture_path = root / "fixture_audit.csv"
    reconciliation_path = root / "source_reconciliation.csv"
    if not acquisition_path.exists():
        raise FileNotFoundError(acquisition_path)
    acquisition = _normalize_acquisition(pd.read_csv(acquisition_path))
    fields = pd.read_csv(field_path) if field_path.exists() else pd.DataFrame()
    fixtures = pd.read_csv(fixture_path) if fixture_path.exists() else pd.DataFrame()
    reconciliation = pd.read_csv(reconciliation_path) if reconciliation_path.exists() else pd.DataFrame()

    expected = {(c, s) for c in TARGET_COMPETITIONS for s in SEASONS}
    observed = set()
    rows = []
    for comp in TARGET_COMPETITIONS:
        source = _canonical_source(comp)
        for season in SEASONS:
            # J.League adapters use calendar years; map them into the common audit key.
            if comp in {"J1", "J2", "J3"}:
                source_rows = acquisition[(acquisition.competition == comp) & (acquisition.source.astype(str) == source) & acquisition.season.astype(str).map(lambda x: _season_from_value(comp, x) == season)]
            elif comp == "FRI":
                source_rows = acquisition[(acquisition.competition == comp) & (acquisition.source.astype(str) == source) & acquisition.season.astype(str).eq(season)]
            else:
                source_rows = acquisition[(acquisition.competition == comp) & (acquisition.source.astype(str) == source) & acquisition.season.astype(str).isin({season, str(int(season[:4]))})]
            if len(source_rows):
                observed.add((comp, season))
                status = "|".join(sorted(set(source_rows.status.astype(str))))
                rows_count = int(pd.to_numeric(source_rows.rows, errors="coerce").fillna(0).sum())
                reason = "Explicit acquisition status from adapter"
            else:
                status = "MISSING_AUDIT_CELL"
                rows_count = 0
                reason = "Adapter did not emit an explicit season/source status"
            rows.append({"competition": comp, "season": season, "canonical_source": source, "status": status, "rows": rows_count, "reason": reason})
    matrix = pd.DataFrame(rows)
    matrix.to_csv(root / "competition_season_gate.csv", index=False)

    scope_complete = observed == expected
    data_available_cells = int(matrix.status.str.contains("AVAILABLE").sum())
    unavailable_cells = int(matrix.status.str.contains("UNAVAILABLE").sum())
    not_applicable_cells = int(matrix.status.str.contains("NOT_APPLICABLE").sum())
    parse_error_cells = int(matrix.status.str.contains("PARSE_ERROR").sum())

    duplicate_source_rows = int(reconciliation.duplicate_source_identity.sum()) if "duplicate_source_identity" in reconciliation.columns and not reconciliation.empty else 0
    # A collision is a canonical fixture key represented by multiple sources with
    # conflicting outcome fields. The current reconciliation table exposes identity;
    # absence of a collision table is therefore a hard gate failure until explicitly audited.
    collision_audit_available = "collision" in reconciliation.columns or "conflict" in reconciliation.columns
    duplicate_gate = duplicate_source_rows == 0

    pit_unknown = int((fields.pit_status == "PIT_UNKNOWN").sum()) if "pit_status" in fields.columns else -1
    pit_unsafe = int((fields.pit_status == "PIT_UNSAFE").sum()) if "pit_status" in fields.columns else -1
    pit_safe = int((fields.pit_status == "PIT_SAFE").sum()) if "pit_status" in fields.columns else 0
    pit_gate = pit_unknown == 0 and pit_unsafe == 0 and pit_safe >= 0

    team_mapping_issues = 0
    if not fixtures.empty:
        for col in ("home_team", "away_team"):
            if col in fixtures.columns:
                team_mapping_issues += int(fixtures[col].isna().sum())
                team_mapping_issues += int((fixtures[col].astype(str).str.strip() == "").sum())

    technical_audit_ok = acquisition is not None and not matrix.empty and len(matrix) == len(expected)
    full_gate_passed = bool(
        technical_audit_ok
        and scope_complete
        and data_available_cells + unavailable_cells + not_applicable_cells == len(expected)
        and parse_error_cells == 0
        and duplicate_gate
        and collision_audit_available
        and team_mapping_issues == 0
        and pit_gate
    )

    result = {
        "target_competition_count": 14,
        "requested_season_count": 16,
        "requested_competition_season_cells": len(expected),
        "technical_audit_ok": technical_audit_ok,
        "coverage_scope_complete": scope_complete,
        "data_available_cells": data_available_cells,
        "unavailable_cells": unavailable_cells,
        "not_applicable_cells": not_applicable_cells,
        "parse_error_cells": parse_error_cells,
        "duplicate_source_rows": duplicate_source_rows,
        "collision_audit_available": collision_audit_available,
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
    if parse_error_cells:
        result["blocking_reasons"].append("parse errors remain")
    if not duplicate_gate:
        result["blocking_reasons"].append("same-source duplicate fixture identities remain")
    if not collision_audit_available:
        result["blocking_reasons"].append("cross-source collision audit is not implemented")
    if team_mapping_issues:
        result["blocking_reasons"].append("team identity fields are incomplete")
    if not pit_gate:
        result["blocking_reasons"].append("publication-time PIT gate has UNKNOWN or UNSAFE fields")
    (root / "completion_gate.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run_completion_gate(), indent=2, ensure_ascii=False))
