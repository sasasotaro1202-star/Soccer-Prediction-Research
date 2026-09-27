import json
from pathlib import Path

from src.research.coverage_funnel_audit import audit


def test_funnel_keeps_downstream_unknown_explicit(tmp_path: Path):
    frontier = tmp_path / "frontier.json"
    frontier.write_text(json.dumps({
        "known_target_event_counts": {"EPL": 12, "J1": 8},
        "discovered_candidates": [
            {"id": "a", "stage": "DISCOVERED"},
            {"id": "b", "stage": "PIT_VALIDATED"},
        ],
    }), encoding="utf-8")
    result = audit(frontier)
    assert result["status"] == "OK"
    assert result["discoverable_known_targets_with_events"] == 2
    assert result["discovered_frontier_candidates"] == 2
    assert result["acquired_targets_explicit"] == 0
    assert result["pit_validated_targets_explicit"] == 0
    assert result["unproven_downstream"]["processed"] is None
    assert result["production_auto_promotion"] is False


def test_funnel_accepts_only_explicit_acquisition_and_pit_evidence(tmp_path: Path):
    frontier = tmp_path / "frontier.json"
    acquisition = tmp_path / "acquisition.json"
    pit = tmp_path / "pit.json"
    frontier.write_text(json.dumps({"known_target_event_counts": {}, "discovered_candidates": []}), encoding="utf-8")
    acquisition.write_text(json.dumps({"acquired_targets": ["EPL", "J1", "EPL"]}), encoding="utf-8")
    pit.write_text(json.dumps({"pit_validated_targets": ["EPL"]}), encoding="utf-8")
    result = audit(frontier, acquisition_path=acquisition, pit_path=pit)
    assert result["acquired_targets_explicit"] == 2
    assert result["pit_validated_targets_explicit"] == 1
