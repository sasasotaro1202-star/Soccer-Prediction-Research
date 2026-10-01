from __future__ import annotations

import pandas as pd

from src.research.experience_learning import (
    _experience_adjusted_probability,
    _teacher_pool,
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


def test_validation_rejects_duplicate_fixture_without_state_identity():
    rows = [
        _row("m1", "2026-01-01T08:00:00Z", "2026-01-01T12:00:00Z", "H"),
        _row("m1", "2026-01-01T09:00:00Z", "2026-01-01T12:00:00Z", "H"),
    ]
    with pytest.raises(RuntimeError, match="duplicate match_id without prediction_state_id"):
        _validate_ledger(pd.DataFrame(rows))


def test_validation_allows_distinct_prediction_states_for_same_fixture():
    rows = [
        _row(
            "m1",
            "2026-01-01T08:00:00Z",
            "2026-01-01T12:00:00Z",
            "H",
            prediction_state_id="state-1",
        ),
        _row(
            "m1",
            "2026-01-01T09:00:00Z",
            "2026-01-01T12:00:00Z",
            "H",
            prediction_state_id="state-2",
        ),
    ]
    validated = _validate_ledger(pd.DataFrame(rows))
    assert len(validated) == 2
    assert set(validated["prediction_state_id"]) == {"state-1", "state-2"}


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


def test_teacher_pool_deduplicates_revised_states_of_one_fixture():
    rows = [
        _row(
            "repeat",
            "2026-01-01T07:00:00Z",
            "2026-01-01T12:00:00Z",
            "H",
            prediction_state_id="repeat-1",
        ),
        _row(
            "repeat",
            "2026-01-01T08:00:00Z",
            "2026-01-01T12:00:00Z",
            "H",
            prediction_state_id="repeat-2",
        ),
    ]
    for i in range(1, 6):
        rows.append(
            _row(
                f"m{i}",
                "2026-01-01T08:00:00Z",
                "2026-01-01T12:00:00Z",
                "H",
                prediction_state_id=f"state-{i}",
            )
        )
    history = _validate_ledger(pd.DataFrame(rows))
    target = history.iloc[0].copy()
    target["prediction_pit_cutoff_utc"] = pd.Timestamp("2026-01-01T13:00:00Z")
    pool = _teacher_pool(history, target)

    assert len(pool) == 6
    assert pool["match_id"].is_unique
    repeat = pool.loc[pool["match_id"] == "repeat"].iloc[0]
    assert repeat["prediction_state_id"] == "repeat-2"


def test_teacher_outcome_after_target_cutoff_is_excluded():
    from src.research.experience_learning import _teacher_pool

    history = pd.DataFrame(
        [
            _row(
                "teacher",
                "2026-01-01T08:00:00Z",
                "2026-01-01T12:00:00Z",
                "H",
            )
        ]
    )
    history = _validate_ledger(
        history.assign(
            kickoff_utc="2026-01-01T10:00:00Z",
        )
    )
    target = history.iloc[0].copy()
    target["prediction_pit_cutoff_utc"] = pd.Timestamp("2026-01-01T11:00:00Z")
    assert _teacher_pool(history, target).empty


def test_chronological_blocks_require_minimum_rows():
    from src.research.experience_learning import _chronological_blocks

    assert len(_chronological_blocks(pd.DataFrame({"x": range(89)}))) == 0
    blocks = _chronological_blocks(pd.DataFrame({"x": range(90)}))
    assert len(blocks) == 90
    assert len(set(blocks.tolist())) == 3


def test_learn_builds_candidate_from_chronological_matured_experience(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.csv"
    policy = tmp_path / "policy.json"
    metrics = tmp_path / "metrics.csv"
    status = tmp_path / "status.json"
    monkeypatch.setattr("src.research.experience_learning.POLICY", policy)
    monkeypatch.setattr("src.research.experience_learning.METRICS", metrics)
    monkeypatch.setattr("src.research.experience_learning.STATUS", status)

    rows = []
    for i in range(90):
        day = pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(days=i)
        rows.append(
            _row(
                f"m{i}",
                (day + pd.Timedelta(hours=8)).isoformat(),
                (day + pd.Timedelta(hours=11)).isoformat(),
                "H",
                kickoff_utc=(day + pd.Timedelta(hours=10)).isoformat(),
                p_home=0.45,
                p_draw=0.30,
                p_away=0.25,
            )
        )
    pd.DataFrame(rows).to_csv(ledger, index=False)

    result = learn(ledger)

    assert result["rows"] == 90
    assert result["blocks"] == 3
    assert result["status"] in {"PROMOTION_CANDIDATE", "HOLD"}
    assert policy.is_file()
    assert metrics.is_file()
    assert status.is_file()

def test_validation_rejects_impossible_experience_timestamp():
    frame = pd.DataFrame(
        [
            _row(
                "m1",
                "2026-01-01T08:00:00Z",
                "2026-01-01T09:00:00Z",
                "H",
            )
        ]
    )
    assert _validate_ledger(frame).empty

