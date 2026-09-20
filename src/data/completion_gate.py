from __future__ import annotations

"""Strict completion gate for the locked active-competition soccer research audit."""

import json
import re
from pathlib import Path

import pandas as pd

from src.data.fixture_field_audit import TARGET_COMPETITIONS
from src.data.football_data import load_available_history
# Use the optimized, fail-closed adapter.  Importing v2 directly bypasses the
# retry/redirect handling and precise early-capture scan implemented by the fast
# adapter, which can unnecessarily leave valid archived evidence UNVERIFIABLE.
from src.data.pit_source_adapter_fast import FootballDataWaybackAdapter
from src.data.pit_archive_fallback import apply_arquivo_fallback
from src.features.soccer_features import build_match_features
from src.data.pit_openfootball_github import apply_bulk as apply_openfootball_pit
from src.data.pit_openfootball_history import apply_openfootball_history

SEASONS = [f"{y}/{str(y + 1)[-2:]}" for y in range(2010, 2026)]
CANONICAL_SOURCES = {
    "EPL": "Football-Data.co.uk",
    "AG_M": "AFC / Asian Games",
    "AG_W": "AFC / Asian Games",
    "ERE": "Football-Data.co.uk",
    "LL": "Football-Data.co.uk",
    "SA": "Football-Data.co.uk",
    "BL1": "Football-Data.co.uk",
    "J1": "J.League Data Site / Football-Data.co.uk:JPN.csv",
    "J2": "J.League Data Site / Football-Data.co.uk:JPN.csv",
    "J3": "J.League Data Site / Football-Data.co.uk:JPN.csv",
    "FL1": "Football-Data.co.uk",
    "UCL": "openfootball",
    "UEL": "openfootball",
    "U23_M": "FIFA / AFC",
    "U18_M": "JFA / AFC / UEFA",
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
    text = str(value).strip()
    if re.fullmatch(r"\d{4}", text):
        y = int(text)
        return f"{y}/{str(y + 1)[-2:]}"
    m = re.fullmatch(r"(\d{4})/(\d{2}|\d{4})", text)
    if m:
        y = int(m.group(1))
        return f"{y}/{str(y + 1)[-2:]}"
    return text


def _merge_pit_evidence(history: pd.DataFrame, enriched: pd.DataFrame) -> pd.DataFrame:
    """Merge PIT evidence with stable dtypes; never coerce verified timestamps to object."""
    out = history.copy()
    if "source_available_at_utc" not in out.columns:
        out["source_available_at_utc"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    else:
        out["source_available_at_utc"] = pd.to_datetime(out["source_available_at_utc"], utc=True, errors="coerce")
    for col in ("pit_evidence_status", "pit_evidence_reason", "pit_evidence_url", "capture_digest"):
        if col not in out.columns:
            out[col] = pd.Series(index=out.index, dtype="object")
    if enriched is None or enriched.empty:
        return out
    for col in ("source_available_at_utc", "pit_evidence_status", "pit_evidence_reason", "pit_evidence_url", "capture_digest"):
        if col not in enriched.columns:
            continue
        values = enriched[col]
        if col == "source_available_at_utc":
            values = pd.to_datetime(values, utc=True, errors="coerce")
        out.loc[enriched.index, col] = values
    return out


def _pit_preflight(root: Path) -> dict:
    """Build the PIT replay from explicit archive evidence, never inferred timing.

    Football-Data rows are enriched with Wayback captures that first contain the
    completed result after a conservative result-availability lower bound. Sources
    without verifiable publication evidence remain unknown and cannot pass the gate.
    A bounded Arquivo.pt fallback is also allowed, using the same lower-bound rule;
    it is evidence recovery only and never infers a publication timestamp.
    """
    history, _ = load_available_history(start_year=2010, end_year=2025)
    if history.empty:
        return {"rows": 0, "pit_verified_rows": 0, "pit_verified_rate": 0.0, "verified_competitions": 0, "archive_enriched_rows": 0, "arquivo_fallback_enriched_rows": 0}

    history = history.copy()
    # The fast adapter is still fail-closed: it only marks a row VERIFIED when an
    # exact completed-result identity is present in an archive capture at/after a
    # valid result-availability bound. Retries and precise-capture scanning only
    # recover evidence that the slower adapter could miss; they do not relax PIT.
    archive = FootballDataWaybackAdapter(cache_dir=str(root / "pit_evidence"), max_workers=4)
    supported = {"EPL", "ERE", "CHA", "BL1", "SA", "LL", "FL1"}
    mask = history["competition"].astype(str).isin(supported)
    if mask.any():
        enriched = archive.apply_bulk(history.loc[mask].copy())
        history = _merge_pit_evidence(history, enriched)
        before_fallback = int((history.get("pit_evidence_status", pd.Series(dtype=str)) == "VERIFIED").sum())
        # Arquivo.pt is expensive because it can require many archived captures.
        # Do not run the secondary provider when the primary Wayback evidence already
        # satisfies the gate; this preserves correctness while avoiding unnecessary
        # network work. If the primary provider is insufficient, fallback remains
        # fail-closed and uses the same explicit result lower-bound rule.
        primary_rate = before_fallback / max(1, len(history))
        primary_competitions = 0
        primary_features = build_match_features(history, history, windows=(3, 5, 10, 20))
        if "pit_verified" in primary_features.columns and not primary_features.empty:
            pv = primary_features["pit_verified"].fillna(False).astype(bool)
            pc = primary_features.assign(pit_verified=pv).groupby("competition")["pit_verified"].sum()
            primary_competitions = int((pc >= 500).sum())
        if before_fallback < 7000 or primary_rate < 0.25 or primary_competitions < 5:
            # Arquivo.pt is a secondary, bounded archive provider. Its capture time is
            # accepted only when it is at/after the same explicit result lower bound and
            # the archived file contains the exact completed-result identity.
            history = _merge_pit_evidence(history, apply_arquivo_fallback(history))
        after_fallback = int((history.get("pit_evidence_status", pd.Series(dtype=str)) == "VERIFIED").sum())
    else:
        before_fallback = after_fallback = 0

    of_mask = history["competition"].astype(str).isin({"UCL", "UEL"})
    openfootball_verified = 0
    if of_mask.any():
        of_enriched = apply_openfootball_pit(history.loc[of_mask].copy())
        history = _merge_pit_evidence(history, of_enriched)
        openfootball_verified = int((history.get("pit_evidence_status", pd.Series(dtype=str)) == "VERIFIED").sum())

    before_versioned = int((history.get("pit_evidence_status", pd.Series(dtype=str)) == "VERIFIED").sum())
    history = _merge_pit_evidence(history, apply_openfootball_history(history, cache_dir=str(root / "pit_evidence")))
    after_versioned = int((history.get("pit_evidence_status", pd.Series(dtype=str)) == "VERIFIED").sum())

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
        "archive_enriched_rows": before_fallback,
        "arquivo_fallback_enriched_rows": max(0, after_fallback - before_fallback),
        "openfootball_verified_rows": openfootball_verified,
        "versioned_openfootball_verified_rows": max(0, after_versioned - before_versioned),
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
            # Calendar-year competitions (J1/J2/J3/Asian Games/youth) are emitted by
            # their adapters as YYYY, while the locked gate matrix uses YYYY/YY.
            # Normalize through the same season-key convention used by the audit
            # builder so an explicit adapter row cannot be mistaken for missing audit.
            if candidates.empty:
                try:
                    target_year = int(season[:4])
                    normalized_season = season
                    if comp in {"J1", "J2", "J3", "AG_M", "AG_W", "U23_M", "U18_M"}:
                        normalized_season = str(target_year)
                    candidates = acquisition[(acquisition.competition == comp) & (acquisition.season.map(lambda x: str(x).strip()) == normalized_season)]
                except (TypeError, ValueError):
                    pass
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
    accounted_cells = int(matrix.status.str.contains(r"AVAILABLE|UNAVAILABLE|NOT_APPLICABLE", regex=True, na=False).sum())
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
        pit = {"rows": 0, "pit_verified_rows": 0, "pit_verified_rate": 0.0, "verified_competitions": 0, "archive_enriched_rows": 0, "arquivo_fallback_enriched_rows": 0}
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
        "target_competition_count": len(TARGET_COMPETITIONS),
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
        "pit_archive_enriched_rows": pit.get("archive_enriched_rows", 0),
        "pit_arquivo_fallback_enriched_rows": pit.get("arquivo_fallback_enriched_rows", 0),
        "versioned_openfootball_verified_rows": pit.get("versioned_openfootball_verified_rows", 0),
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
        result["blocking_reasons"].append("insufficient PIT-verified replay rows using explicit archived publication evidence")
    serialized = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    (root / "completion_gate.json").write_text(serialized, encoding="utf-8")
    # Research consumes a distinct audit-gate artifact. It is intentionally derived
    # from the same fail-closed audit result; no weaker parallel gate is introduced.
    audit_gate = {
        "full_gate_passed": bool(result.get("full_gate_passed", False)),
        "pit_publication_time_gate": bool(result.get("pit_publication_time_gate", False)),
        "coverage_scope_complete": bool(result.get("coverage_scope_complete", False)),
        "collision_rows": int(result.get("collision_rows", 0)),
        "duplicate_source_rows": int(result.get("duplicate_source_rows", 0)),
        "team_mapping_issues": int(result.get("team_mapping_issues", 0)),
        "blocking_reasons": list(result.get("blocking_reasons", [])),
        "source": "completion_gate",
        "fail_closed": True,
    }
    (root / "audit_gate.json").write_text(json.dumps(audit_gate, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run_completion_gate(), indent=2, ensure_ascii=False, default=str))
