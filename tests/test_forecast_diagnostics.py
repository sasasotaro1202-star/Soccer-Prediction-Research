from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.prediction.model_bundle import load_bundle, train_and_save_bundle
from src.research.forecast_diagnostics import OUTPUT_COLUMNS, build


def _training():
    rng = np.random.default_rng(7)
    n = 90
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    target = (f1 + 0.2 * f2 > 0).astype(int)
    target[:3] = [0, 1, 2]
    return pd.DataFrame({
        "f1": f1,
        "f2": f2,
        "kickoff_utc": pd.date_range("2026-01-01", periods=n, freq="h"),
        "pit_verified": True,
        "target": target,
    })


def _fixtures():
    d = _training().iloc[:5].copy()
    d["match_id"] = [f"m{i}" for i in range(len(d))]
    d["competition"] = "EPL"
    d["home_team"] = "Home"
    d["away_team"] = "Away"
    return d


def test_forecast_diagnostics_are_outcome_free_and_bounded(tmp_path):
    df = _training()
    path = tmp_path / "bundle.pkl"
    train_and_save_bundle(
        df,
        ["f1", "f2"],
        {"weights": {"logistic": 1.0}, "temperature": 1.0},
        str(path),
        "diag-test",
        "snapshot",
    )
    bundle = load_bundle(str(path))
    result = build(_fixtures()[["match_id", "kickoff_utc", "competition", "home_team", "away_team", "f1", "f2"]], bundle)

    assert tuple(result.columns) == OUTPUT_COLUMNS
    assert len(result) == 5
    assert "target" not in result.columns
    numeric = result[list(OUTPUT_COLUMNS[5:16])].to_numpy(dtype=float)
    assert np.all(np.isfinite(numeric))
    assert np.all((numeric >= 0.0) & (numeric <= 1.0))
    assert set(result["routing_risk_bucket"]) <= {"LOW", "MEDIUM", "HIGH"}
    assert result["routing_route"].notna().all()


def test_forecast_diagnostics_reject_missing_features(tmp_path):
    df = _training()
    path = tmp_path / "bundle.pkl"
    train_and_save_bundle(
        df,
        ["f1", "f2"],
        {"weights": {"logistic": 1.0}, "temperature": 1.0},
        str(path),
        "diag-test",
        "snapshot",
    )
    bundle = load_bundle(str(path))
    fixtures = _fixtures()[["match_id", "kickoff_utc", "competition", "home_team", "away_team", "f1"]]
    with pytest.raises(RuntimeError, match="missing"):
        build(fixtures, bundle)



def test_forecast_diagnostics_reject_invalid_kickoff(tmp_path):
    df = _training()
    path = tmp_path / "bundle.pkl"
    train_and_save_bundle(
        df,
        ["f1", "f2"],
        {"weights": {"logistic": 1.0}, "temperature": 1.0},
        str(path),
        "diag-test",
        "snapshot",
    )
    bundle = load_bundle(str(path))
    fixtures = _fixtures()[[
        "match_id", "kickoff_utc", "competition", "home_team", "away_team", "f1", "f2"
    ]].copy()
    fixtures.loc[0, "kickoff_utc"] = "not-a-timestamp"
    with pytest.raises(RuntimeError, match="invalid kickoff_utc"):
        build(fixtures, bundle)
