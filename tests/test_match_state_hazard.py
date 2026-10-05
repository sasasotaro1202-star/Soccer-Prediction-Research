import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.research.match_state_hazard import (
    EVENT_TYPES,
    build_state_features,
    fit_hazard_model,
    predict_hazard,
    propagate_scenarios,
    run_match_state_research,
    validate_snapshot_contract,
)


def _rows(matches: int = 24) -> pd.DataFrame:
    base = pd.Timestamp("2026-01-01T12:00:00Z")
    rows = []
    for i in range(matches):
        kickoff = base + pd.Timedelta(days=i)
        final_away = 1 if i % 5 == 0 else 0
        final_home = 2 if i % 2 else 1
        for minute in (5, 10, 15, 20, 25):
            cutoff = kickoff + pd.Timedelta(minutes=minute)
            if minute == 5 and i % 2 == 0:
                event = "HOME_GOAL"
                next_time = cutoff + pd.Timedelta(minutes=5)
            elif minute == 10 and i % 5 == 0:
                event = "AWAY_GOAL"
                next_time = cutoff + pd.Timedelta(minutes=5)
            elif minute == 15 and i % 7 == 0:
                event = "HOME_RED"
                next_time = cutoff + pd.Timedelta(minutes=5)
            else:
                event = "NO_EVENT"
                next_time = pd.NaT
            rows.append(
                {
                    "match_id": f"m{i}",
                    "kickoff_utc": kickoff,
                    "prediction_cutoff_utc": cutoff,
                    "event_time_utc": cutoff,
                    "source_available_at_utc": cutoff - pd.Timedelta(seconds=30),
                    "pit_verified": True,
                    "home_score": 1 if minute >= 10 and i % 2 == 0 else 0,
                    "away_score": 1 if minute >= 15 and i % 5 == 0 else 0,
                    "home_red_cards": 1 if minute >= 20 and i % 7 == 0 else 0,
                    "away_red_cards": 0,
                    "next_event_type": event,
                    "next_event_time_utc": next_time,
                    "hazard_window_minutes": 5.0,
                    "label_available_at_utc": kickoff + pd.Timedelta(minutes=110),
                    "final_home_goals": final_home,
                    "final_away_goals": final_away,
                    "home_elo": 1520 + i,
                    "away_elo": 1490,
                    "elo_diff": 30 + i,
                }
            )
    return pd.DataFrame(rows)


def test_validate_snapshot_contract_rejects_post_cutoff_source():
    frame = _rows(1)
    frame.loc[0, "source_available_at_utc"] = pd.Timestamp(
        "2026-01-01T12:05:01Z"
    )
    with pytest.raises(ValueError, match="source availability"):
        validate_snapshot_contract(frame)


def test_validate_snapshot_contract_rejects_label_available_before_cutoff():
    frame = _rows(1)
    frame.loc[0, "label_available_at_utc"] = frame.loc[0, "prediction_cutoff_utc"]
    with pytest.raises(ValueError, match="label is available"):
        validate_snapshot_contract(frame)


def test_state_features_are_deterministic_and_endogenous():
    frame = validate_snapshot_contract(_rows(2))
    out = build_state_features(frame)
    assert np.allclose(out["score_diff"], out["home_score"] - out["away_score"])
    assert np.all(out["remaining_minutes"] >= 0)
    assert {
        "elapsed_minute",
        "remaining_minutes",
        "score_diff",
        "score_total",
        "late_game",
    }.issubset(out.columns)


def test_hazard_model_returns_fixed_event_order():
    frame = build_state_features(validate_snapshot_contract(_rows(30)))
    model = fit_hazard_model(
        frame,
        as_of_cutoff=pd.Timestamp("2026-02-15T00:00:00Z"),
        min_rows=50,
    )
    assert model["research_only"] is True
    assert model["event_types"] == list(EVENT_TYPES)
    probs = predict_hazard(model, frame.iloc[0].to_dict())
    assert probs.shape == (len(EVENT_TYPES),)
    assert np.isfinite(probs).all()
    assert np.all(probs > 0)
    assert np.isclose(probs.sum(), 1.0)


def test_scenario_propagation_is_normalized_and_reproducible():
    frame = build_state_features(validate_snapshot_contract(_rows(30)))
    model = fit_hazard_model(
        frame,
        as_of_cutoff=pd.Timestamp("2026-02-15T00:00:00Z"),
        min_rows=50,
    )
    state = frame.iloc[0].to_dict()
    first = propagate_scenarios(model, state, horizon_minutes=15, step_minutes=5)
    second = propagate_scenarios(model, state, horizon_minutes=15, step_minutes=5)
    assert first == second
    outcome = first["outcome_probabilities"]
    assert np.isclose(sum(outcome.values()), 1.0)
    assert first["residual_mass_before_normalization"] > 0
    assert first["simulation"]["method"] == "deterministic_scenario_propagation"


def test_run_is_fail_closed_warmup_without_dataset(tmp_path):
    status = run_match_state_research(
        tmp_path / "missing.csv",
        tmp_path / "artifacts",
    )
    assert status["status"] == "WARMUP"
    assert status["production_usable"] is False
    assert status["oos_claimed"] is False
    saved = (tmp_path / "artifacts" / "match_state_status.json").read_text()
    assert '"research_only": true' in saved


def test_chronological_hazard_oos_blocks_immature_labels_at_test_cutoff():
    frame = build_state_features(validate_snapshot_contract(_rows(150)))
    base = pd.Timestamp("2026-01-01T12:00:00Z")
    # Put many matches close together and delay final-label maturity so a naive
    # "max training label" as-of boundary would reach into the future test block.
    kickoff_by_match = {
        f"m{i}": base + pd.Timedelta(minutes=2 * i) for i in range(150)
    }
    # Shift every time field consistently so the fixture remains contract-valid.
    old_kickoff = frame["kickoff_utc"].copy()
    cutoff_offset = frame["prediction_cutoff_utc"] - old_kickoff
    event_offset = frame["event_time_utc"] - old_kickoff
    next_event_offset = frame["next_event_time_utc"] - old_kickoff
    frame["kickoff_utc"] = frame["match_id"].map(kickoff_by_match)
    frame["prediction_cutoff_utc"] = frame["kickoff_utc"] + cutoff_offset
    frame["event_time_utc"] = frame["kickoff_utc"] + event_offset
    frame["next_event_time_utc"] = frame["kickoff_utc"] + next_event_offset
    frame["source_available_at_utc"] = frame["prediction_cutoff_utc"] - pd.Timedelta(seconds=30)
    frame["label_available_at_utc"] = frame["kickoff_utc"] + pd.Timedelta(minutes=120)
    result = __import__(
        "src.research.match_state_hazard",
        fromlist=["evaluate_hazard_chronological_oos"],
    ).evaluate_hazard_chronological_oos(
        frame,
        method="logistic",
        min_training_matches=20,
        min_test_matches=10,
        max_folds=4,
    )
    for fold in result.get("folds", []):
        assert pd.Timestamp(fold["training_label_cutoff_utc"]) <= pd.Timestamp(
            fold["test_start_utc"]
        )
    if result["status"] == "EVALUATED":
        assert result["overall_next_event_logloss"] == np.mean(
            [f["next_event_logloss"] for f in result["folds"]]
        )


def test_chronological_hazard_oos_is_match_level_and_research_only():
    frame = build_state_features(validate_snapshot_contract(_rows(150)))
    from src.research.match_state_hazard import evaluate_hazard_chronological_oos
    result = evaluate_hazard_chronological_oos(
        frame,
        method="logistic",
        min_training_matches=60,
        min_test_matches=10,
        max_folds=2,
    )
    assert result["status"] == "EVALUATED"
    assert result["oos_claimed"] is True
    assert result["production_usable"] is False
    assert result["policy"].startswith("match-level_expanding_chronological_OOS")
    assert result["snapshot_rows"] > 0
    assert np.isfinite(result["overall_next_event_logloss"])


def test_match_state_workflow_is_autonomous_but_non_production():
    workflow = Path(".github/workflows/soccer-match-state-research.yml").read_text(encoding="utf-8")
    assert "actions: write" in workflow
    assert 'gh workflow run "adaptive_data_discovery.yml"' in workflow
    assert 'gh workflow run "soccer-source-probe.yml"' in workflow
    assert "performance_verified_false" in workflow
    assert "promotion_candidate_false" in workflow
    assert "|| true" not in workflow


def test_match_state_workflow_can_persist_research_state_but_not_pr_state():
    workflow = Path(".github/workflows/soccer-match-state-research.yml").read_text(encoding="utf-8")
    assert "contents: write" in workflow
    assert "github.event_name != 'pull_request'" in workflow
    assert "Persist research status on main" in workflow
