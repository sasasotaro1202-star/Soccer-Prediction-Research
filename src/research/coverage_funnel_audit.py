"""Coverage-funnel audit for the soccer research scope.

The funnel is deliberately conservative. Missing downstream evidence is
reported as UNKNOWN/0 observed rather than inferred as successful acquisition.
This mirrors the cross-project coverage audits and keeps discovery separate
from production eligibility.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data.competition_sources import TARGET_COMPETITIONS


_STAGE_ORDER = (
    "DISCOVERED",
    "METADATA_CHECKED",
    "DATA_FEASIBLE",
    "PIT_VALIDATED",
    "SHADOW",
    "OOS/ROBUSTNESS_VALIDATED",
    "LIMITED_PRODUCTION",
    "STABLE_PRODUCTION",
    "SCALE-UP",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _candidate_stage_counts(frontier: dict[str, Any]) -> dict[str, int]:
    counts = {stage: 0 for stage in _STAGE_ORDER}
    candidates = frontier.get("discovered_candidates")
    if not isinstance(candidates, list):
        return counts
    for item in candidates:
        if not isinstance(item, dict):
            continue
        stage = str(item.get("stage", "DISCOVERED"))
        if stage in counts:
            counts[stage] += 1
    return counts


def audit(frontier_path: str | Path, *, acquisition_path: str | Path | None = None,
          pit_path: str | Path | None = None) -> dict[str, Any]:
    frontier = _read_json(Path(frontier_path))
    stage_counts = _candidate_stage_counts(frontier)

    known_counts = frontier.get("known_target_event_counts")
    known_counts = known_counts if isinstance(known_counts, dict) else {}
    known_active = {str(k): int(v) for k, v in known_counts.items() if int(v) > 0}

    acquisition = _read_json(Path(acquisition_path)) if acquisition_path else {}
    pit = _read_json(Path(pit_path)) if pit_path else {}

    acquisition_known = acquisition.get("acquired_targets")
    if not isinstance(acquisition_known, list):
        acquisition_known = []
    pit_validated = pit.get("pit_validated_targets")
    if not isinstance(pit_validated, list):
        pit_validated = []

    eligible = len(TARGET_COMPETITIONS)
    discovered_known = len(known_active)
    frontier_discovered = sum(stage_counts.values())
    acquired = len({str(x) for x in acquisition_known if x})
    validated = len({str(x) for x in pit_validated if x})

    result = {
        "schema_version": 1,
        "status": "OK" if frontier else "NO_FRONTIER_EVIDENCE",
        "definitions": {
            "eligible": "Strict active target competition definitions.",
            "discoverable": "Publicly observed/mapped target identities.",
            "discovered": "Known active observations plus research frontier candidates.",
            "acquired": "Only explicitly supplied acquisition evidence; never inferred from discovery.",
            "pit_validated": "Only explicitly supplied PIT evidence; never inferred from acquisition.",
            "processed": "Not inferred by this audit.",
            "outcome_confirmed": "Not inferred by this audit.",
        },
        "eligible_targets": eligible,
        "discoverable_known_targets_with_events": discovered_known,
        "discovered_frontier_candidates": frontier_discovered,
        "discovered_total_observed": discovered_known + frontier_discovered,
        "acquired_targets_explicit": acquired,
        "pit_validated_targets_explicit": validated,
        "stage_counts": stage_counts,
        "known_target_event_counts": dict(sorted(known_active.items())),
        "unproven_downstream": {
            "processed": None,
            "outcome_confirmed": None,
        },
        "coverage_rates": {
            "known_event_target_rate": float(discovered_known / eligible) if eligible else 0.0,
            "discovery_over_eligible_ratio": float((discovered_known + frontier_discovered) / eligible) if eligible else 0.0,
            "explicit_acquisition_rate": float(acquired / eligible) if eligible else None,
            "explicit_pit_validation_rate": float(validated / eligible) if eligible else None,
        },
        "production_auto_promotion": False,
        "fail_closed_on_unknown_downstream_state": True,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier", required=True)
    parser.add_argument("--acquisition", default=None)
    parser.add_argument("--pit", default=None)
    parser.add_argument("--output", default="artifacts/coverage_funnel.json")
    args = parser.parse_args()
    result = audit(args.frontier, acquisition_path=args.acquisition, pit_path=args.pit)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "eligible_targets": result["eligible_targets"],
        "discovered_total_observed": result["discovered_total_observed"],
        "acquired_targets_explicit": result["acquired_targets_explicit"],
        "pit_validated_targets_explicit": result["pit_validated_targets_explicit"],
    }, ensure_ascii=False))
    return 0 if result["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
