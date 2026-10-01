from __future__ import annotations

import pandas as pd

from src.research.experience_learning import (
    _experience_adjusted_probability,
    _validate_ledger,
    learn,
)


def _row(match_id: str, prediction: str, available: str, actual: str, **extra):
    row = {
        "match_id": match_id,
        "kickoff_utc": "2026-01-01T10:00:00Z",
        "prediction_pit_cutoff_utc": prediction,
        "prediction_pit_gate": "PASS",
        "experience_available_at_utc": available,
        "competition": "EPL",
        "model_version": "v1",
        "p_home": 0.60,
        "p_draw": 0.20,
        "p_away": 0.20,
        "actual_result": actual,
    }
    row.update(extra)
    return row


def test_validation_rejects_teacher_available_before_teacher_prediction():
    frame = pd.DataFrame(
        [_row("m1", "2026-01-01T08:00:00Z", "2026-01-01T07:00:00Z", "H")]
    )
    assert _validate_ledger(frame).empty


def test_experience_correction_is_probability_valid():
    baseline = pd.Series({"p_home": 0.6, "p_draw": 0.2, "p_away": 0.2})
    teacher = pd.DataFrame(
        {
            "actual_result": ["H", "H", "D", "A", "H"],
        }
    )
    adjusted, weight, source, n = _experience_adjusted_probability(
        baseline.to_numpy(dtype=float), teacher, 20.0
    )
    assert 0 < weight < 1
    assert source == "unknown"
    assert n == 5
    assert (adjusted > 0).all()
    assert abs(float(adjusted.sum()) - 1.0) < 1e-9


def test_learn_fails_closed_without_pit_valid_settled_experience(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.csv"
    policy = tmp_path / "policy.json"
    metrics = tmp_path / "metrics.csv"
    status = tmp_path / "status.json"
    monkeypatch.setattr("src.research.experience_learning.POLICY", policy)
    monkeypatch.setattr("src.research.experience_learning.METRICS", metrics)
    monkeypatch.setattr("src.research.experience_learning.STATUS", status)
    pd.DataFrame(
        [_row("m1", "2026-01-01T08:00:00Z", "2026-01-01T07:00:00Z", "H")]
    ).to_csv(ledger, index=False)
    result = learn(ledger)
    assert result["status"] == "INSUFFICIENT_EXPERIENCE"
    assert policy.is_file()
    assert status.is_file()
