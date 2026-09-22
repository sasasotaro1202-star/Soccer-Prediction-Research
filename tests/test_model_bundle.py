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
