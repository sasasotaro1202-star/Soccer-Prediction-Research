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


def test_bundle_preserves_and_applies_contextual_weights(tmp_path):
    df = _fixture().copy()
    df["competition"] = "EPL"
    df["elo_diff"] = 0.0
    df["home_goal_total_avg_5"] = 2.0
    df["away_goal_total_avg_5"] = 2.0
    df["home_draw_rate_20"] = 0.25
    df["away_draw_rate_20"] = 0.25
    df["rest_diff_hours"] = 0.0
    df["neutral_venue_known"] = True
    df["neutral_venue"] = False
    feature_cols = ["f1", "f2"]
    path = tmp_path / "production_model.pkl"
    selection = {
        "weights": {"logistic": 0.5, "hist_gb": 0.5},
        "temperature": 1.0,
        "context_weights": {
            "FULL:EPL|EVEN|MID_LOW|MID_LOW|EVEN|HOME_AWAY": {
                "logistic": 1.0,
                "hist_gb": 0.0,
            }
        },
        "contextual_temperatures": {
            "EPL|EVEN|MID_LOW|MID_LOW|EVEN|HOME_AWAY": 1.0,
        },
    }
    train_and_save_bundle(
        df,
        feature_cols,
        selection,
        str(path),
        "test-context-version",
        "snapshot-context",
    )
    bundle = load_bundle(str(path))
    assert "context_weights" in bundle
    assert "contextual_temperatures" in bundle

    x = df[feature_cols + [
        "competition", "elo_diff", "home_goal_total_avg_5",
        "away_goal_total_avg_5", "home_draw_rate_20", "away_draw_rate_20",
        "rest_diff_hours", "neutral_venue_known", "neutral_venue",
    ]].iloc[:8]
    predicted = predict_bundle(bundle, x)
    logistic = bundle["models"]["logistic"].predict_proba(x[feature_cols])
    assert np.allclose(predicted, logistic, atol=1e-7)
