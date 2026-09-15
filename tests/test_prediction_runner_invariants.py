import pandas as pd
import pytest

from src.prediction.runner import _eligible_fixtures, _normalize_prediction_time


def _fixture_rows():
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "kickoff_utc": "2026-09-16T12:00:00Z",
                "home_team": "A",
                "away_team": "B",
                "source_available_at_utc": "2026-09-15T10:00:00Z",
                "pit_verified": True,
                "starter_status": "ANNOUNCED",
            }
        ]
    )


def test_prediction_time_normalizes_naive_and_aware_timestamps():
    assert _normalize_prediction_time("2026-09-15T10:00:00").tzinfo is not None
    assert _normalize_prediction_time("2026-09-15T19:00:00+09:00").isoformat() == "2026-09-15T10:00:00+00:00"


def test_duplicate_match_ids_fail_closed():
    rows = pd.concat([_fixture_rows(), _fixture_rows()], ignore_index=True)
    with pytest.raises(RuntimeError, match="duplicate match_id"):
        _eligible_fixtures(rows, _normalize_prediction_time("2026-09-15T09:00:00Z"))


def test_empty_team_identity_fails_closed():
    rows = _fixture_rows()
    rows.loc[0, "home_team"] = "   "
    with pytest.raises(RuntimeError, match="home_team"):
        _eligible_fixtures(rows, _normalize_prediction_time("2026-09-15T09:00:00Z"))


def test_future_fixture_requires_source_availability_by_prediction_time():
    rows = _fixture_rows()
    rows.loc[0, "source_available_at_utc"] = "2026-09-15T11:00:00Z"
    eligible = _eligible_fixtures(rows, _normalize_prediction_time("2026-09-15T10:00:00Z"))
    assert eligible.empty


def test_invalid_required_timestamps_fail_closed():
    rows = _fixture_rows()
    rows.loc[0, "kickoff_utc"] = "not-a-timestamp"
    with pytest.raises(RuntimeError, match="kickoff_utc"):
        _eligible_fixtures(rows, _normalize_prediction_time("2026-09-15T09:00:00Z"))

    rows = _fixture_rows()
    rows.loc[0, "source_available_at_utc"] = "not-a-timestamp"
    with pytest.raises(RuntimeError, match="source_available_at_utc"):
        _eligible_fixtures(rows, _normalize_prediction_time("2026-09-15T09:00:00Z"))


def test_missing_starter_status_fails_closed():
    rows = _fixture_rows()
    rows.loc[0, "starter_status"] = ""
    with pytest.raises(RuntimeError, match="starter_status"):
        _eligible_fixtures(rows, _normalize_prediction_time("2026-09-15T09:00:00Z"))


def test_unverified_or_unannounced_fixture_is_excluded():
    rows = _fixture_rows()
    rows.loc[0, "pit_verified"] = False
    assert _eligible_fixtures(rows, _normalize_prediction_time("2026-09-15T09:00:00Z")).empty

    rows = _fixture_rows()
    rows.loc[0, "starter_status"] = "EXPECTED"
    assert _eligible_fixtures(rows, _normalize_prediction_time("2026-09-15T09:00:00Z")).empty


def test_fixture_at_prediction_time_is_not_future():
    rows = _fixture_rows()
    rows.loc[0, "kickoff_utc"] = "2026-09-15T10:00:00Z"
    assert _eligible_fixtures(rows, _normalize_prediction_time("2026-09-15T10:00:00Z")).empty


def test_prediction_runner_rejects_wrong_probability_shape(tmp_path, monkeypatch):
    from src.prediction import runner

    monkeypatch.setattr(runner, "load_adopted_model", lambda path: {"adoption_status": "ADOPT", "model_version": "v1"})
    monkeypatch.setattr(runner, "load_bundle", lambda path: {"model_version": "v1"})
    monkeypatch.setattr(runner, "predict_bundle", lambda bundle, X: [[0.6, 0.4]])

    fixtures_path = tmp_path / "future.csv"
    _fixture_rows().to_csv(fixtures_path, index=False)
    with pytest.raises(RuntimeError, match="unexpected probability shape"):
        runner.run(
            fixtures_path=str(fixtures_path),
            bundle_path=str(tmp_path / "model.pkl"),
            output_path=str(tmp_path / "predictions.csv"),
            status_path=str(tmp_path / "status.json"),
            prediction_time="2026-09-15T09:00:00Z",
            registry_path=str(tmp_path / "registry.json"),
        )


def test_prediction_runner_rejects_nonfinite_probabilities(tmp_path, monkeypatch):
    from src.prediction import runner

    monkeypatch.setattr(runner, "load_adopted_model", lambda path: {"adoption_status": "ADOPT", "model_version": "v1"})
    monkeypatch.setattr(runner, "load_bundle", lambda path: {"model_version": "v1"})
    monkeypatch.setattr(runner, "predict_bundle", lambda bundle, X: [[float("nan"), 0.3, 0.7]])

    fixtures_path = tmp_path / "future.csv"
    _fixture_rows().to_csv(fixtures_path, index=False)
    with pytest.raises(RuntimeError, match="invalid probabilities"):
        runner.run(
            fixtures_path=str(fixtures_path),
            bundle_path=str(tmp_path / "model.pkl"),
            output_path=str(tmp_path / "predictions.csv"),
            status_path=str(tmp_path / "status.json"),
            prediction_time="2026-09-15T09:00:00Z",
            registry_path=str(tmp_path / "registry.json"),
        )
