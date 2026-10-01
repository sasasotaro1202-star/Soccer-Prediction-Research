from __future__ import annotations

import pandas as pd
import pytest

from src.research.experience_candidates import build_experience_candidate_plan


def _row(i: int, *, confidence: float = 0.8, disagreement=None, uncertainty=None, routing_risk=None):
    p_home = confidence
    p_draw = (1.0 - confidence) / 2.0
    p_away = p_draw
    return {
        "match_id": f"m{i}",
        "prediction_state_id": f"s{i}",
        "kickoff_utc": f"2026-01-{(i % 28) + 1:02d}T10:00:00Z",
        "prediction_pit_cutoff_utc": f"2026-01-{(i % 28) + 1:02d}T08:00:00Z",
        "prediction_pit_gate": "PASS",
        "experience_available_at_utc": f"2026-01-{(i % 28) + 1:02d}T12:00:00Z",
        "competition": "EPL" if i < 30 else "UCL",
        "model_version": "v1",
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "actual_result": "D" if i >= 30 else "H",
        "shadow_model_disagreement": disagreement,
        "shadow_uncertainty_score": uncertainty,
        "shadow_routing_risk": routing_risk,
    }


def test_empty_experience_is_warmup():
    plan = build_experience_candidate_plan(pd.DataFrame())
    assert plan["status"] == "WARMUP"
    assert plan["candidates"] == []
    assert plan["safety_contract"]["research_only"] is True


def test_requires_matured_outcome_after_kickoff():
    rows = [_row(i) for i in range(30)]
    rows[0]["experience_available_at_utc"] = "2026-01-01T09:00:00Z"
    plan = build_experience_candidate_plan(pd.DataFrame(rows))
    assert plan["settled_rows"] == 29


def test_generates_segment_candidate_only_when_degraded():
    rows = [_row(i) for i in range(30)]
    for i in range(30):
        rows[i]["competition"] = "EPL"
        rows[i]["actual_result"] = "H"
    rows.extend(_row(30 + i) for i in range(30))
    plan = build_experience_candidate_plan(pd.DataFrame(rows))
    assert plan["status"] == "READY"
    assert any(c["source_dimension"] == "competition" and c["source_segment"] == "UCL" for c in plan["candidates"])


def test_high_confidence_wrong_candidate_uses_error_telemetry():
    rows = [_row(i, disagreement=0.01) for i in range(60)]
    for i, row in enumerate(rows):
        row["actual_result"] = "D" if i < 30 else "H"
        row["shadow_model_disagreement"] = 0.30 if i < 30 else 0.01
    plan = build_experience_candidate_plan(pd.DataFrame(rows))
    assert plan["status"] == "READY"
    assert any(c["source_segment"] == "high_confidence_wrong" for c in plan["candidates"])


def test_high_disagreement_candidate_uses_shadow_telemetry():
    rows = [_row(i, disagreement=0.01) for i in range(60)]
    for i, row in enumerate(rows):
        if i < 30:
            row["actual_result"] = "D"
            row["shadow_model_disagreement"] = 0.30
        else:
            row["actual_result"] = "H"
            row["shadow_model_disagreement"] = 0.01
    plan = build_experience_candidate_plan(pd.DataFrame(rows))
    assert plan["status"] == "READY"
    assert any(c["source_segment"] == "high_model_disagreement" for c in plan["candidates"])


def test_duplicate_fixture_without_state_identity_fails_closed():
    rows = [_row(i) for i in range(30)]
    rows[1]["match_id"] = rows[0]["match_id"]
    rows[0].pop("prediction_state_id")
    rows[1].pop("prediction_state_id")
    with pytest.raises(RuntimeError, match="duplicate match_id without prediction_state_id"):
        build_experience_candidate_plan(pd.DataFrame(rows))


def test_candidate_plan_is_research_only():
    rows = [_row(i) for i in range(60)]
    plan = build_experience_candidate_plan(pd.DataFrame(rows))
    assert plan["safety_contract"]["production_changed"] is False
    assert plan["safety_contract"]["frozen_holdout_allowed"] is False
    assert all(candidate["research_only"] for candidate in plan["candidates"])

def test_missing_shadow_telemetry_is_safe():
    rows = [_row(i) for i in range(60)]
    for row in rows:
        row.pop("shadow_model_disagreement", None)
        row.pop("shadow_uncertainty_score", None)
        row.pop("shadow_routing_risk", None)
    plan = build_experience_candidate_plan(pd.DataFrame(rows))
    assert plan["status"] in {"READY", "WARMUP"}
    assert all(candidate["research_only"] for candidate in plan["candidates"])

def test_mature_history_without_degradation_is_not_warmup():
    rows = [_row(i) for i in range(60)]
    for row in rows:
        row["actual_result"] = "H"
    plan = build_experience_candidate_plan(pd.DataFrame(rows))
    assert plan["status"] == "NO_ACTIONABLE_DEGRADATION"
