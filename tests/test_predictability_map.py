from __future__ import annotations

import pandas as pd

from src.research.predictability_map import _predictability_score, analyze


def _row(i: int, *, risk: float = 0.1, wrong: bool = False) -> dict:
    p = [0.70, 0.20, 0.10] if not wrong else [0.15, 0.70, 0.15]
    return {
        "match_id": f"m{i}",
        "prediction_state_id": f"s{i}",
        "kickoff_utc": f"2026-01-{1 + (i % 28):02d}T12:00:00Z",
        "prediction_pit_cutoff_utc": f"2026-01-{1 + (i % 28):02d}T10:00:00Z",
        "experience_available_at_utc": f"2026-01-{1 + (i % 28):02d}T14:00:00Z",
        "prediction_pit_gate": "PASS",
        "p_home": p[0],
        "p_draw": p[1],
        "p_away": p[2],
        "actual_result": "A" if wrong else "H",
        "predictive_entropy": risk,
        "model_disagreement": risk,
        "covariate_drift": risk,
        "history_support_risk": risk,
        "routing_risk": risk,
    }


def test_predictability_score_is_bounded_and_degrades_with_risk():
    scored = _predictability_score(
        pd.DataFrame([_row(0, risk=0.05), _row(1, risk=0.95)])
    )
    assert 0.0 <= float(scored.loc[0, "predictability_score"]) <= 1.0
    assert 0.0 <= float(scored.loc[1, "predictability_score"]) <= 1.0
    assert float(scored.loc[0, "predictability_score"]) > float(
        scored.loc[1, "predictability_score"]
    )


def test_high_confidence_low_predictability_is_marked():
    row = _row(0, risk=0.95, wrong=True)
    row["p_home"] = 0.80
    row["p_draw"] = 0.15
    row["p_away"] = 0.05
    scored = _predictability_score(pd.DataFrame([row]))
    assert bool(scored.loc[0, "high_confidence_low_predictability"]) is True


def test_missing_telemetry_is_not_imputed_into_a_false_high_score():
    row = _row(0, risk=0.1)
    for col in ("predictive_entropy", "model_disagreement", "covariate_drift"):
        row.pop(col)
    scored = _predictability_score(pd.DataFrame([row]))
    assert pd.isna(scored.loc[0, "predictability_score"])
    assert scored.loc[0, "predictability_band"] == "UNKNOWN"


def test_mature_outcomes_are_required_and_duplicate_states_reduce_to_latest_fixture():
    rows = [_row(i, risk=0.2, wrong=(i % 5 == 0)) for i in range(40)]
    duplicate = dict(rows[0])
    duplicate["prediction_state_id"] = "s-later"
    duplicate["prediction_pit_cutoff_utc"] = "2026-01-01T11:00:00Z"
    rows.append(duplicate)
    state = analyze(pd.DataFrame(rows))
    assert state["rows"] == 40
    assert state["production_usable"] is False
    assert state["safety_contract"]["outcome_data_used_only_after_maturity"] is True


def test_chronological_failure_risk_candidate_is_research_only():
    rows = []
    for i in range(420):
        block_risk = 0.10 if i < 180 else 0.85
        wrong = i >= 180 and i % 2 == 0
        rows.append(_row(i, risk=block_risk, wrong=wrong))
    frame = pd.DataFrame(rows)
    frame["kickoff_utc"] = pd.date_range(
        "2026-01-01", periods=len(frame), freq="h", tz="UTC"
    )
    frame["prediction_pit_cutoff_utc"] = (
        frame["kickoff_utc"] - pd.Timedelta(hours=2)
    )
    frame["experience_available_at_utc"] = (
        frame["kickoff_utc"] + pd.Timedelta(hours=2)
    )
    state = analyze(frame)
    assert state["rows"] == 420
    assert len(state["oos_blocks"]) >= 3
    assert state["production_usable"] is False
    assert state["safety_contract"]["production_probabilities_changed"] is False
    assert state["safety_contract"]["frozen_holdout_touched"] is False


def test_empty_ledger_is_safe_warmup():
    state = analyze(pd.DataFrame())
    assert state["status"] == "WARMUP"
    assert state["rows"] == 0
    assert state["production_usable"] is False
    assert state["safety_contract"]["production_changed"] is False


def test_shadow_prefixed_telemetry_is_consumed():
    row = _row(0, risk=0.2)
    for name in (
        "predictive_entropy",
        "model_disagreement",
        "covariate_drift",
        "history_support_risk",
        "routing_risk",
    ):
        row["shadow_" + name] = row.pop(name)
    scored = _predictability_score(
        pd.DataFrame([row])
        .assign(
            predictive_entropy=lambda d: d["shadow_predictive_entropy"],
            model_disagreement=lambda d: d["shadow_model_disagreement"],
            covariate_drift=lambda d: d["shadow_covariate_drift"],
            history_support_risk=lambda d: d["shadow_history_support_risk"],
            routing_risk=lambda d: d["shadow_routing_risk"],
        )
    )
    assert pd.notna(scored.loc[0, "predictability_score"])


def test_duplicate_fixture_without_state_identity_fails_closed():
    rows = [_row(i, risk=0.2) for i in range(60)]
    rows[1]["match_id"] = rows[0]["match_id"]
    rows[0].pop("prediction_state_id")
    rows[1].pop("prediction_state_id")
    try:
        analyze(pd.DataFrame(rows))
    except RuntimeError as exc:
        assert any(token in str(exc) for token in ("duplicate match_id without prediction_state_id", "invalid prediction_state_id"))
    else:
        raise AssertionError("duplicate fixture without state identity must fail closed")


def test_predictability_oos_training_respects_maturity_cutoff():
    from src.research.pit_training import filter_prior_mature_training

    frame = pd.DataFrame({
        "prediction_pit_cutoff_utc": [
            "2026-01-01T08:00:00Z",
            "2026-01-01T09:00:00Z",
            "2026-01-01T10:00:00Z",
        ],
        "experience_available_at_utc": [
            "2026-01-01T08:30:00Z",
            "2026-01-01T10:30:00Z",
            "2026-01-01T09:30:00Z",
        ],
    })
    eligible = filter_prior_mature_training(frame, pd.Timestamp("2026-01-01T10:00:00Z"))
    assert len(eligible) == 2
