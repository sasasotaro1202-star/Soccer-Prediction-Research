import numpy as np
import pandas as pd
import pytest

from src.research.experience_candidate_evaluator import (
    _fingerprint,
    evaluate_experience_candidates,
)


def _cases():
    rows = []
    for block in range(5):
        start = pd.Timestamp("2024-01-01T00:00:00Z") + pd.Timedelta(days=block * 30)
        for i in range(40):
            idx = block * 40 + i
            confidence = 0.85 if i < 20 else 0.55
            actual = 0 if i < 25 else 1
            p_home = 0.75 if actual == 0 else 0.20
            p_draw = 0.15
            p_away = 1 - p_home - p_draw
            rows.append({
                "match_id": f"m{idx}",
                "oos_start": start,
                "competition": "EPL" if i < 30 else "J1",
                "kickoff_utc": start + pd.Timedelta(minutes=i),
                "actual": actual,
                "correct": int((np.argmax([p_home, p_draw, p_away]) == actual)),
                "p_home": p_home,
                "p_draw": p_draw,
                "p_away": p_away,
                "confidence": confidence,
                "model_disagreement": 0.30 if i < 15 else 0.05,
                "uncertainty_score": 0.70 if i < 15 else 0.10,
                "risk_score": 0.60 if i < 15 else 0.10,
            })
    return pd.DataFrame(rows)


def _plan(*candidates):
    return {
        "schema_version": 1,
        "candidates": list(candidates),
        "safety_contract": {
            "research_only": True,
            "production_changed": False,
            "frozen_holdout_allowed": False,
        },
    }


def _candidate(dimension, segment, cid):
    return {
        "candidate_id": cid,
        "source_dimension": dimension,
        "source_segment": segment,
        "evidence_n": 100,
        "impact_vs_global_logloss": 0.10,
        "research_only": True,
        "production_changed": False,
        "frozen_holdout_allowed": False,
    }


def test_fingerprint_is_stable_and_order_independent():
    d = _cases().iloc[:40].copy()
    assert _fingerprint(d) == _fingerprint(d.sample(frac=1, random_state=7))


def test_scoped_evaluator_excludes_last_two_locked_blocks():
    cases = _cases()
    blocks = (
        cases[["oos_start"]]
        .drop_duplicates()
        .sort_values("oos_start")
        .assign(
            oos_end=lambda x: x["oos_start"] + pd.Timedelta(minutes=39),
            n=40,
        )
    )
    plan = _plan(
        _candidate("competition", "EPL", "c1"),
        _candidate("confidence_bucket", "0.8-0.9", "c2"),
        _candidate("error_types", "high_model_disagreement", "c3"),
        _candidate("model", "some-model", "c4"),
    )
    result = evaluate_experience_candidates(plan, cases, blocks)
    assert result["policy"]["locked_holdout_used"] is False
    assert result["policy"]["diagnostic_reuse_only"] is True
    assert result["candidate_count"] == 4
    by_id = {x["candidate_id"]: x for x in result["evaluations"]}
    assert by_id["c4"]["status"] == "NO_DIRECT_OOS_SCOPE"
    assert by_id["c1"]["folds"] >= 3
    assert by_id["c2"]["folds"] >= 3
    assert by_id["c3"]["folds"] >= 3
    assert by_id["c1"]["next_action"] in {"CREATE_FRESH_CHRONOLOGICAL_OOS_CANDIDATE", "HOLD"}


def test_unsafe_plan_fails_closed():
    cases = _cases()
    plan = _plan(_candidate("competition", "EPL", "unsafe"))
    plan["safety_contract"]["production_changed"] = True
    with pytest.raises(ValueError, match="production mutation"):
        evaluate_experience_candidates(plan, cases)


def test_insufficient_scope_is_explicit():
    cases = _cases()
    plan = _plan(_candidate("competition", "NONEXISTENT", "none"))
    result = evaluate_experience_candidates(plan, cases)
    row = result["evaluations"][0]
    assert row["status"] == "NO_SCOPE_ROWS"
    assert row["promotion_allowed"] is False
    assert row["fresh_oos_required"] is True
