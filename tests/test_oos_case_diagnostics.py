import numpy as np
import pandas as pd

import src.evaluation.walk_forward as wf


class DummyModel:
    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        n = len(X)
        return np.tile(np.asarray([[0.5, 0.3, 0.2]], dtype=float), (n, 1))


def test_run_walk_forward_writes_per_match_risk_diagnostics(monkeypatch, tmp_path):
    monkeypatch.setattr(wf, "candidates", lambda random_state=42: {"logistic": DummyModel()})
    n = 180
    kickoff = pd.date_range("2024-01-01T00:00:00Z", periods=n, freq="6h")
    target = np.resize(np.asarray([0, 1, 2], dtype=int), n)
    frame = pd.DataFrame({
        "match_id": [f"m{i}" for i in range(n)],
        "kickoff_utc": kickoff,
        "competition": ["TEST"] * n,
        "season_start": [2024] * n,
        "target": target,
        "pit_verified": [True] * n,
        "prediction_cutoff_at_utc": kickoff - pd.Timedelta(minutes=60),
        "feature_source_max_available_at_utc": kickoff - pd.Timedelta(minutes=120),
        "source_available_at_utc": kickoff + pd.Timedelta(hours=3),
        "x": np.linspace(-1.0, 1.0, n),
    })
    output = tmp_path / "oos_case_diagnostics.csv"

    metrics, selection = wf.run_walk_forward(
        frame,
        ["x"],
        min_train=140,
        oos_block=30,
        case_output_path=str(output),
    )

    assert not metrics.empty
    assert not selection.empty
    assert output.is_file()
    cases = pd.read_csv(output)
    required = {
        "match_id", "competition", "kickoff_utc", "actual", "prediction",
        "correct", "p_home", "p_draw", "p_away", "confidence", "margin",
        "risk_score", "risk_bucket",
        "model_disagreement", "predictive_entropy", "uncertainty_score",
        "covariate_drift", "history_support_risk", "routing_route",
        "prediction_cutoff_at_utc", "feature_source_max_available_at_utc",
        "pit_verified", "outcome_source_available_at_utc",
        "pit_lineage_status", "pit_lineage_hash",
    }
    assert required.issubset(cases.columns)
    assert len(cases) == int(metrics["n"].sum())
    assert cases["match_id"].is_unique
    assert cases["risk_bucket"].isin({"LOW", "MEDIUM", "HIGH"}).all()
    for column in (
        "model_disagreement", "predictive_entropy", "uncertainty_score",
        "covariate_drift", "history_support_risk",
    ):
        assert np.isfinite(cases[column]).all()
        assert ((cases[column] >= 0.0) & (cases[column] <= 1.0)).all()
    assert cases["routing_route"].notna().all()
    assert np.allclose(cases[["p_home", "p_draw", "p_away"]].sum(axis=1), 1.0)

    assert cases["pit_lineage_status"].eq("PASS").all()
    assert cases["pit_lineage_hash"].str.startswith("pit:").all()
    cutoff = pd.to_datetime(cases["prediction_cutoff_at_utc"], utc=True)
    available = pd.to_datetime(cases["feature_source_max_available_at_utc"], utc=True)
    kickoff_values = pd.to_datetime(cases["kickoff_utc"], utc=True)
    assert (available <= cutoff).all()
    assert (kickoff_values > cutoff).all()
