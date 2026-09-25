import json

import numpy as np
import pandas as pd
import pytest

from src.models.mom_model import (
    MOM_FEATURE_COLUMNS,
    fit_mom_model,
    predict_mom_distribution,
    select_mom_top4_from_distribution,
    MissingnessAwareMOMTransformer,
)


def _rows(n_matches=6):
    rows = []
    base = pd.Timestamp("2025-01-01", tz="UTC")
    for m in range(n_matches):
        kickoff = base + pd.Timedelta(days=m)
        for p in range(6):
            rows.append(
                {
                    "match_id": f"m{m}",
                    "player_id": f"p{m}_{p}",
                    "kickoff_utc": kickoff,
                    "feature_available_at_utc": kickoff - pd.Timedelta(hours=1),
                    "pit_verified": True,
                    "is_motm": int(p == 0),
                    "recent_rating_ewm": 7.5 - p * 0.1 + m * 0.01,
                    "recent_minutes_ewm": 80 - p,
                    "recent_goals_per90_ewm": 0.2 + (5 - p) * 0.02,
                    "recent_assists_per90_ewm": 0.1 + (5 - p) * 0.01,
                    "recent_xg_per90_ewm": 0.15 + (5 - p) * 0.02,
                    "recent_xa_per90_ewm": 0.10 + (5 - p) * 0.01,
                    "recent_key_passes_per90_ewm": 1.0 + (5 - p) * 0.2,
                    "recent_shots_per90_ewm": 2.0 + (5 - p) * 0.25,
                    "recent_starts_rate": 0.8 - p * 0.03,
                    "team_attack_strength": 1.0 + m * 0.01,
                    "opponent_defense_strength": 1.0,
                    "position_attack_weight": 1.0 if p < 4 else 0.7,
                    "days_rest": 6.0,
                }
            )
    return pd.DataFrame(rows)


def test_fit_uses_only_pit_verified_rows_and_records_metadata():
    d = _rows(6)
    d.loc[d["match_id"] == "m5", "pit_verified"] = False
    model = fit_mom_model(d, min_matches=5)
    assert model["metadata"]["training_matches"] == 5
    assert model["metadata"]["training_rows"] == 30
    assert tuple(model["metadata"]["feature_columns"]) == MOM_FEATURE_COLUMNS


def test_fit_supports_hist_gbdt_research_challenger():
    model = fit_mom_model(_rows(8), min_matches=5, method="hist_gbdt")
    candidates = _rows(1).drop(columns=["is_motm"])
    dist = predict_mom_distribution(
        model,
        candidates,
        prediction_time=pd.Timestamp("2025-01-01T00:00:00Z") - pd.Timedelta(minutes=1),
    )
    assert len(dist) == 6
    assert np.isfinite(dist["probability"]).all()
    assert float(dist["probability"].sum()) == pytest.approx(1.0, abs=1e-10)


def test_prediction_is_full_distribution_and_top4_is_exactly_four():
    model = fit_mom_model(_rows(6), min_matches=5)
    candidates = _rows(1).drop(columns=["is_motm"])
    dist = predict_mom_distribution(
        model,
        candidates,
        prediction_time=pd.Timestamp("2025-01-01T00:00:00Z") - pd.Timedelta(minutes=1),
    )
    assert len(dist) == 6
    assert dist["probability"].between(0, 1).all()
    assert float(dist["probability"].sum()) == pytest.approx(1.0, abs=1e-10)
    top4 = select_mom_top4_from_distribution(dist)
    assert len(top4) == 4
    assert [x.rank for x in top4] == [1, 2, 3, 4]


def test_prediction_normalizes_probabilities_independently_per_fixture():
    model = fit_mom_model(_rows(6), min_matches=5)
    one = _rows(1).drop(columns=["is_motm"])
    two = _rows(1).drop(columns=["is_motm"]).copy()
    two["match_id"] = "m_other"
    two["kickoff_utc"] = two["kickoff_utc"] + pd.Timedelta(days=1)
    two["feature_available_at_utc"] = two["kickoff_utc"] - pd.Timedelta(hours=1)
    candidates = pd.concat([one, two], ignore_index=True)
    dist = predict_mom_distribution(model, candidates, prediction_time=None)
    sums = dist.groupby("match_id")["probability"].sum()
    assert set(sums.index) == {"m0", "m_other"}
    assert all(abs(float(x) - 1.0) < 1e-10 for x in sums)


def test_prediction_for_a_fixture_is_batch_invariant():
    model = fit_mom_model(_rows(6), min_matches=5)
    one = _rows(1).drop(columns=["is_motm"])
    other = _rows(1).drop(columns=["is_motm"]).copy()
    other["match_id"] = "m_other"
    other["kickoff_utc"] = other["kickoff_utc"] + pd.Timedelta(days=1)
    other["feature_available_at_utc"] = other["kickoff_utc"] - pd.Timedelta(hours=1)
    pt = "2024-12-31T23:00:00Z"

    first = predict_mom_distribution(model, one, prediction_time=None)
    batched = predict_mom_distribution(model, pd.concat([one, other], ignore_index=True), prediction_time=None)
    a = first.sort_values("player_id")["probability"].to_numpy()
    b = batched.loc[batched["match_id"] == "m0"].sort_values("player_id")["probability"].to_numpy()
    assert np.allclose(a, b, atol=1e-12)


def test_top4_selection_refuses_multiple_fixtures():
    model = fit_mom_model(_rows(6), min_matches=5)
    candidates = _rows(1).drop(columns=["is_motm"])
    other = candidates.copy()
    other["match_id"] = "m_other"
    other["kickoff_utc"] = other["kickoff_utc"] + pd.Timedelta(days=1)
    other["feature_available_at_utc"] = other["kickoff_utc"] - pd.Timedelta(hours=1)
    dist = predict_mom_distribution(model, pd.concat([candidates, other], ignore_index=True), prediction_time=None)
    with pytest.raises(ValueError, match="exactly one match_id"):
        select_mom_top4_from_distribution(dist)


def test_prediction_rejects_outcome_labels():
    model = fit_mom_model(_rows(6), min_matches=5)
    with pytest.raises(ValueError, match="must not contain future outcome"):
        predict_mom_distribution(
            model,
            _rows(1),
            prediction_time="2024-12-31T23:59:00Z",
        )


def test_prediction_fails_closed_when_feature_was_available_after_kickoff():
    model = fit_mom_model(_rows(6), min_matches=5)
    candidates = _rows(1).drop(columns=["is_motm"])
    candidates.loc[0, "feature_available_at_utc"] = candidates.loc[0, "kickoff_utc"] + pd.Timedelta(minutes=1)
    with pytest.raises(RuntimeError, match="PIT violation"):
        predict_mom_distribution(model, candidates)


def test_prediction_fails_closed_when_feature_is_future_at_prediction_time():
    model = fit_mom_model(_rows(6), min_matches=5)
    candidates = _rows(1).drop(columns=["is_motm"])
    candidates.loc[:, "feature_available_at_utc"] = pd.Timestamp("2024-12-31T23:45:00Z")
    with pytest.raises(RuntimeError, match="feature unavailable"):
        predict_mom_distribution(
            model,
            candidates,
            prediction_time="2024-12-31T23:30:00Z",
        )


def test_all_missing_feature_is_not_fabricated_and_is_tracked():
    d = _rows(8)
    d["recent_goals_per90_ewm"] = np.nan
    transformer = MissingnessAwareMOMTransformer()
    x = transformer.fit_transform(d[list(MOM_FEATURE_COLUMNS)])
    assert "recent_goals_per90_ewm" in transformer.inactive_feature_columns_
    assert x.shape[1] == len(MOM_FEATURE_COLUMNS) * 2 - 1
    assert np.isfinite(x).all()

def test_missing_feature_values_are_imputed_without_zeroing():
    d = _rows(8)
    d.loc[d["match_id"] == "m0", "recent_goals_per90_ewm"] = np.nan
    d.loc[d["match_id"] == "m1", "recent_shots_per90_ewm"] = np.nan
    model = fit_mom_model(d, min_matches=5)
    candidates = _rows(1).drop(columns=["is_motm"])
    candidates.loc[0, "recent_goals_per90_ewm"] = np.nan
    candidates.loc[1, "recent_shots_per90_ewm"] = np.nan
    dist = predict_mom_distribution(model, candidates, prediction_time=None)
    assert len(dist) == 6
    assert np.isfinite(dist["probability"]).all()
    assert float(dist["probability"].sum()) == pytest.approx(1.0, abs=1e-10)
    assert model["metadata"]["missingness_aware_imputation"] is True


def test_training_rejects_match_without_exactly_one_motm():
    d = _rows(5)
    d.loc[d["match_id"] == "m0", "is_motm"] = 0
    with pytest.raises(ValueError, match="exactly one positive label"):
        fit_mom_model(d, min_matches=5)


def test_unknown_pit_rows_do_not_get_imputed_as_safe():
    d = _rows(5)
    d["pit_verified"] = False
    with pytest.raises(RuntimeError, match="No PIT-verified"):
        fit_mom_model(d, min_matches=5)


def test_conditional_mom_model_uses_fixture_level_choice_structure():
    model = fit_mom_model(
        _rows(8),
        min_matches=5,
        method="conditional_logit",
        regularization_c=0.30,
    )
    assert model["metadata"]["method"] == "pit_player_form_conditional_logit"
    candidates = _rows(1).drop(columns=["is_motm"])
    dist = predict_mom_distribution(
        model,
        candidates,
        prediction_time="2024-12-31T23:00:00Z",
    )
    assert len(dist) == 6
    assert float(dist["probability"].sum()) == pytest.approx(1.0, abs=1e-10)


def test_feature_names_are_outcome_free():
    forbidden = {"motm", "player_of_the_match", "final", "result", "match_rating"}
    assert not any(any(token in c.lower() for token in forbidden) for c in MOM_FEATURE_COLUMNS)


def test_dataset_rating_proxy_label_is_explicit_and_outcome_only():
    from scripts.run_mom_research import _build_dataset_rating_proxy_labels

    player_matches = pd.DataFrame([
        {"fixture_id": 10, "player_id": 1, "player_name": "A", "rating": 7.1, "minutes": 90},
        {"fixture_id": 10, "player_id": 2, "player_name": "B", "rating": 7.8, "minutes": 80},
        {"fixture_id": 10, "player_id": 3, "player_name": "C", "rating": 7.8, "minutes": 70},
    ])
    target_fixtures = pd.DataFrame([{"id": 10}])
    labels = _build_dataset_rating_proxy_labels(player_matches, target_fixtures)
    assert labels.loc[0, "label_status"] == "LABEL_FOUND"
    assert labels.loc[0, "label_source"] == "dataset_rating_top_performer_proxy"
    assert labels.loc[0, "player_id"] == "2"
    assert labels.loc[0, "event_id"] == "dataset-performance-proxy:10"


def test_dataset_performance_proxy_uses_flat_stats_when_rating_is_missing():
    import numpy as np
    from scripts.run_mom_research import _build_dataset_rating_proxy_labels

    player_matches = pd.DataFrame([
        {"fixture_id": 20, "player_id": 11, "player_name": "A", "rating": float("nan"), "minutes": 90},
        {"fixture_id": 20, "player_id": 12, "player_name": "B", "rating": float("nan"), "minutes": 80},
    ])
    player_stats = pd.DataFrame([
        {"fixture_id": 20, "player_id": 11, "games_rating": float("nan"), "games_minutes": 90,
         "goals_total": 0, "goals_assists": 1, "passes_key": 4, "shots_total": 2},
        {"fixture_id": 20, "player_id": 12, "games_rating": float("nan"), "games_minutes": 80,
         "goals_total": 1, "goals_assists": 0, "passes_key": 1, "shots_total": 3},
    ])
    labels = _build_dataset_rating_proxy_labels(
        player_matches, pd.DataFrame([{"id": 20}]), player_stats=player_stats
    )
    assert labels.loc[0, "player_id"] == "12"
    assert labels.loc[0, "label_source"] == "dataset_boxscore_top_performer_proxy"
