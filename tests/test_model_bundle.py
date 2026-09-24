import numpy as np
import pandas as pd
import pytest

from src.prediction.model_bundle import load_bundle, predict_bundle, train_and_save_bundle


def _fixture():
    rng = np.random.default_rng(42)
    n = 120
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    target = (x1 + 0.3 * x2 > 0).astype(int)
    # Keep all three classes represented for multiclass classifiers.
    target[:3] = [0, 1, 2]
    return pd.DataFrame({"kickoff_utc": pd.date_range("2020-01-01", periods=n, freq="h"), "pit_verified": True, "target": target, "f1": x1, "f2": x2})


def test_bundle_roundtrip_and_probability_sums(tmp_path):
    df = _fixture()
    path = tmp_path / "production_model.pkl"
    meta = train_and_save_bundle(
        df,
        ["f1", "f2"],
        {"weights": {"logistic": 1.0}, "temperature": 1.0},
        str(path),
        "test-version",
        "snapshot-1",
    )
    assert meta["fit_rows"] == len(df)
    bundle = load_bundle(str(path))
    probs = predict_bundle(bundle, df[["f1", "f2"]].iloc[:10])
    assert probs.shape == (10, 3)
    assert np.all(np.isfinite(probs))
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_bundle_rejects_missing_features(tmp_path):
    df = _fixture()
    path = tmp_path / "production_model.pkl"
    train_and_save_bundle(df, ["f1", "f2"], {"weights": {"logistic": 1.0}, "temperature": 1.0}, str(path), "test-version", "snapshot-1")
    bundle = load_bundle(str(path))
    with pytest.raises(RuntimeError, match="missing model features"):
        predict_bundle(bundle, df[["f1"]].iloc[:1])


def test_bundle_uses_locked_oos_verified_recency_score_method(tmp_path):
    df = _fixture().copy()
    df["home_team"] = np.where(np.arange(len(df)) % 2 == 0, "A", "B")
    df["away_team"] = np.where(np.arange(len(df)) % 2 == 0, "B", "A")
    df["home_goals"] = np.where(df["target"] == 0, 2, np.where(df["target"] == 1, 1, 0))
    df["away_goals"] = np.where(df["target"] == 2, 2, np.where(df["target"] == 1, 1, 0))
    path = tmp_path / "production_model.pkl"
    train_and_save_bundle(
        df,
        ["f1", "f2"],
        {"weights": {"logistic": 1.0}, "temperature": 1.0},
        str(path),
        "test-version",
        "snapshot-1",
        score_selection={"selected_method": "recency", "status": "ADOPT_CANDIDATE"},
        score_locked_gate={"selected_method": "recency", "status": "PASS"},
    )
    bundle = load_bundle(str(path))
    assert bundle["score_method"] == "recency"
    assert bundle["score_model"]["method"] == "pit_recency_weighted_venue_split_team_goal_rates"


def test_bundle_uses_locked_oos_verified_time_decay_score_method(tmp_path):
    df = _fixture().copy()
    df["home_team"] = np.where(np.arange(len(df)) % 2 == 0, "A", "B")
    df["away_team"] = np.where(np.arange(len(df)) % 2 == 0, "B", "A")
    df["home_goals"] = np.where(df["target"] == 0, 2, np.where(df["target"] == 1, 1, 3))
    df["away_goals"] = np.where(df["target"] == 2, 2, np.where(df["target"] == 1, 1, 0))
    path = tmp_path / "production_model.pkl"
    train_and_save_bundle(
        df,
        ["f1", "f2"],
        {"weights": {"logistic": 1.0}, "temperature": 1.0},
        str(path),
        "test-version",
        "snapshot-1",
        score_selection={"selected_method": "time_decay", "status": "ADOPT_CANDIDATE"},
        score_locked_gate={"selected_method": "time_decay", "status": "PASS"},
    )
    bundle = load_bundle(str(path))
    assert bundle["score_method"] == "time_decay"
    assert bundle["score_model"]["method"] == "pit_time_decay_weighted_venue_split_team_goal_rates"


def test_load_bundle_rejects_invalid_time_decay_half_life(tmp_path):
    df = _fixture().copy()
    df["home_team"] = np.where(np.arange(len(df)) % 2 == 0, "A", "B")
    df["away_team"] = np.where(np.arange(len(df)) % 2 == 0, "B", "A")
    df["home_goals"] = np.where(df["target"] == 0, 2, np.where(df["target"] == 1, 1, 3))
    df["away_goals"] = np.where(df["target"] == 2, 2, np.where(df["target"] == 1, 1, 0))
    path = tmp_path / "production_model.pkl"
    train_and_save_bundle(
        df,
        ["f1", "f2"],
        {"weights": {"logistic": 1.0}, "temperature": 1.0},
        str(path),
        "test-version",
        "snapshot-1",
        score_selection={"selected_method": "time_decay"},
        score_locked_gate={"selected_method": "time_decay", "status": "PASS"},
    )
    with path.open("rb") as fh:
        bundle = __import__("pickle").load(fh)
    bundle["score_model"]["half_life_days"] = 0
    with path.open("wb") as fh:
        __import__("pickle").dump(bundle, fh)
    with pytest.raises(RuntimeError, match="half_life_days"):
        load_bundle(str(path))


def test_bundle_does_not_promote_unverified_score_selection(tmp_path):
    df = _fixture().copy()
    df["home_team"] = np.where(np.arange(len(df)) % 2 == 0, "A", "B")
    df["away_team"] = np.where(np.arange(len(df)) % 2 == 0, "B", "A")
    df["home_goals"] = np.where(df["target"] == 0, 2, np.where(df["target"] == 1, 1, 0))
    df["away_goals"] = np.where(df["target"] == 2, 2, np.where(df["target"] == 1, 1, 0))
    path = tmp_path / "production_model.pkl"
    train_and_save_bundle(
        df,
        ["f1", "f2"],
        {"weights": {"logistic": 1.0}, "temperature": 1.0},
        str(path),
        "test-version",
        "snapshot-1",
        score_selection={"selected_method": "recency", "status": "ADOPT_CANDIDATE"},
        score_locked_gate={"selected_method": "primary", "status": "REJECT"},
    )
    bundle = load_bundle(str(path))
    assert bundle["score_method"] == "primary"


def test_load_bundle_rejects_unverified_non_primary_score_method(tmp_path):
    df = _fixture().copy()
    df["home_team"] = np.where(np.arange(len(df)) % 2 == 0, "A", "B")
    df["away_team"] = np.where(np.arange(len(df)) % 2 == 0, "B", "A")
    df["home_goals"] = np.where(df["target"] == 0, 2, np.where(df["target"] == 1, 1, 0))
    df["away_goals"] = np.where(df["target"] == 2, 2, np.where(df["target"] == 1, 1, 0))
    path = tmp_path / "production_model.pkl"
    train_and_save_bundle(
        df, ["f1", "f2"], {"weights": {"logistic": 1.0}, "temperature": 1.0},
        str(path), "test-version", "snapshot-1",
        score_selection={"selected_method": "recency"},
        score_locked_gate={"selected_method": "primary", "status": "REJECT"},
    )
    with path.open("rb") as fh:
        bundle = __import__("pickle").load(fh)
    bundle["score_method"] = "recency"
    bundle["score_model"]["method"] = "pit_recency_weighted_venue_split_team_goal_rates"
    bundle["score_locked_verification"] = {"selected_method": "recency", "status": "REJECT"}
    with path.open("wb") as fh:
        __import__("pickle").dump(bundle, fh)
    with pytest.raises(RuntimeError, match="requires matching PASS locked-OOS verification"):
        load_bundle(str(path))


def test_bundle_uses_locked_oos_verified_negative_binomial_score_method(tmp_path):
    df = _fixture().copy()
    df["home_team"] = np.where(np.arange(len(df)) % 2 == 0, "A", "B")
    df["away_team"] = np.where(np.arange(len(df)) % 2 == 0, "B", "A")
    df["home_goals"] = np.where(df["target"] == 0, 2, np.where(df["target"] == 1, 1, 3))
    df["away_goals"] = np.where(df["target"] == 2, 2, np.where(df["target"] == 1, 1, 0))
    path = tmp_path / "production_model.pkl"
    train_and_save_bundle(
        df,
        ["f1", "f2"],
        {"weights": {"logistic": 1.0}, "temperature": 1.0},
        str(path),
        "test-version",
        "snapshot-1",
        score_selection={"selected_method": "negative_binomial", "status": "ADOPT_CANDIDATE"},
        score_locked_gate={"selected_method": "negative_binomial", "status": "PASS"},
    )
    bundle = load_bundle(str(path))
    assert bundle["score_method"] == "negative_binomial"
    assert bundle["score_model"]["method"] == "negative_binomial_pit_overdispersed_team_rates"


def test_bundle_uses_locked_oos_verified_neutral_aware_score_method(tmp_path):
    df = _fixture().copy()
    df["home_team"] = np.where(np.arange(len(df)) % 2 == 0, "A", "B")
    df["away_team"] = np.where(np.arange(len(df)) % 2 == 0, "B", "A")
    df["home_goals"] = np.where(df["target"] == 0, 2, np.where(df["target"] == 1, 1, 3))
    df["away_goals"] = np.where(df["target"] == 2, 2, np.where(df["target"] == 1, 1, 0))
    path = tmp_path / "production_model.pkl"
    train_and_save_bundle(
        df,
        ["f1", "f2"],
        {"weights": {"logistic": 1.0}, "temperature": 1.0},
        str(path),
        "test-version",
        "snapshot-1",
        score_selection={"selected_method": "neutral_aware", "status": "ADOPT_CANDIDATE"},
        score_locked_gate={"selected_method": "neutral_aware", "status": "PASS"},
    )
    bundle = load_bundle(str(path))
    assert bundle["score_method"] == "neutral_aware"
    assert bundle["score_model"]["method"] == "neutral_aware_pit_smoothed_venue_split_team_goal_rates"


def test_bundle_preserves_and_applies_validated_contextual_routing(tmp_path):
    df = _fixture().copy()
    df["competition"] = np.where(np.arange(len(df)) % 2 == 0, "EPL", "LL")
    df["elo_diff"] = np.linspace(-150, 150, len(df))
    df["home_goal_total_avg_5"] = 2.2
    df["away_goal_total_avg_5"] = 1.9
    df["home_draw_rate_20"] = 0.28
    df["away_draw_rate_20"] = 0.26
    df["rest_diff_hours"] = 0.0
    df["neutral_venue_known"] = True
    df["neutral_venue"] = False

    path = tmp_path / "production_model.pkl"
    selection = {
        "weights": {"logistic": 0.5, "extra_trees": 0.5},
        "temperature": 1.0,
        "context_weights": {
            "COMP:EPL": {"logistic": 0.0, "extra_trees": 1.0},
            "COMP:LL": {"logistic": 1.0, "extra_trees": 0.0},
        },
        "contextual_temperatures": {
            "COMP:EPL": 1.20,
            "COMP:LL": 0.90,
            "GLOBAL": 1.0,
        },
        "contextual_temperature_reasons": {
            "COMP:EPL": "test_route",
            "COMP:LL": "test_route",
            "GLOBAL": "global_fallback",
        },
    }
    train_and_save_bundle(
        df,
        ["f1", "f2"],
        selection,
        str(path),
        "test-version",
        "snapshot-1",
    )
    bundle = load_bundle(str(path))
    assert bundle["schema_version"] == 3
    assert bundle["routing_policy"]["type"] == "hierarchical_validation_context"
    assert bundle["routing_policy"]["context_weights"]["COMP:EPL"]["extra_trees"] == 1.0

    routed = predict_bundle(bundle, df.iloc[:10])
    legacy = dict(bundle)
    legacy.pop("routing_policy")
    global_only = predict_bundle(legacy, df.iloc[:10])
    assert np.allclose(routed.sum(axis=1), 1.0)
    assert np.all(np.isfinite(routed))
    assert np.max(np.abs(routed - global_only)) > 1e-6


def test_bundle_persists_and_applies_risk_temperature_modifiers(tmp_path):
    df = _fixture().copy()
    df["competition"] = "EPL"
    df["elo_diff"] = np.linspace(-150, 150, len(df))
    df["home_goal_total_avg_5"] = 2.2
    df["away_goal_total_avg_5"] = 1.9
    df["home_draw_rate_20"] = 0.28
    df["away_draw_rate_20"] = 0.26
    df["rest_diff_hours"] = 0.0
    df["neutral_venue_known"] = True
    df["neutral_venue"] = False

    selection = {
        "weights": {"logistic": 1.0},
        "temperature": 1.0,
        "context_weights": {"COMP:EPL": {"logistic": 1.0}},
        "contextual_temperatures": {"COMP:EPL": 1.0, "GLOBAL": 1.0},
        "contextual_temperature_reasons": {"COMP:EPL": "test", "GLOBAL": "test"},
        "risk_temperature_modifiers": {"LOW": 1.0, "MEDIUM": 1.0, "HIGH": 1.20},
        "risk_temperature_reasons": {"HIGH": "test"},
        "dynamic_routing": {
            "schema_version": 1,
            "type": "drift_uncertainty_router",
            "enabled": True,
            "drift_strength": 0.85,
            "uncertainty_strength": 0.75,
            "min_specialist_trust": 0.25,
            "feature_cols": ["f1", "f2"],
        },
    }
    path = tmp_path / "production_model.pkl"
    train_and_save_bundle(
        df,
        ["f1", "f2"],
        selection,
        str(path),
        "test-version",
        "snapshot-1",
    )
    bundle = load_bundle(str(path))
    assert bundle["routing_policy"]["risk_temperature_modifiers"]["HIGH"] == 1.20

    extreme = df[["f1", "f2"]].iloc[:1].copy()
    extreme.loc[:, "f1"] = 50.0
    extreme.loc[:, "f2"] = 50.0
    probs = predict_bundle(bundle, extreme)
    assert probs.shape == (1, 3)
    assert np.all(np.isfinite(probs))
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_bundle_defaults_unvalidated_matchday_policy_to_shadow_only(tmp_path):
    df = _fixture()
    path = tmp_path / "production_model.pkl"
    train_and_save_bundle(
        df, ["f1", "f2"], {"weights": {"logistic": 1.0}, "temperature": 1.0},
        str(path), "test-version", "snapshot-1",
    )
    bundle = load_bundle(str(path))
    assert bundle["matchday_policy"]["schema_version"] == 1
    assert bundle["matchday_policy"]["status"] == "SHADOW_ONLY"


def test_bundle_rejects_matchday_policy_pass_without_oos_evidence(tmp_path):
    df = _fixture()
    path = tmp_path / "production_model.pkl"
    train_and_save_bundle(
        df, ["f1", "f2"],
        {
            "weights": {"logistic": 1.0},
            "temperature": 1.0,
            "matchday_policy": {"schema_version": 1, "status": "PASS"},
        },
        str(path), "test-version", "snapshot-1",
    )
    with pytest.raises(RuntimeError, match="requires explicit OOS evidence"):
        load_bundle(str(path))


def test_bundle_accepts_matchday_policy_pass_with_locked_oos_evidence(tmp_path):
    df = _fixture()
    path = tmp_path / "production_model.pkl"
    policy = {
        "schema_version": 1,
        "status": "PASS",
        "evidence": {
            "oos_verified": True,
            "locked_holdout_untouched": True,
        },
    }
    train_and_save_bundle(
        df, ["f1", "f2"],
        {
            "weights": {"logistic": 1.0},
            "temperature": 1.0,
            "matchday_policy": policy,
        },
        str(path), "test-version", "snapshot-1",
    )
    bundle = load_bundle(str(path))
    assert bundle["matchday_policy"] == policy
