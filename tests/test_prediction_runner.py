from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.prediction.runner import _eligible_fixtures


def _fixtures(tmp_path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "ok",
                "kickoff_utc": "2026-09-15T18:00:00Z",
                "home_team": "A",
                "away_team": "B",
                "competition": "EPL",
                "source_available_at_utc": "2026-09-14T00:00:00Z",
                "pit_verified": True,
                "starter_status": "ANNOUNCED",
            },
            {
                "match_id": "late-info",
                "kickoff_utc": "2026-09-15T19:00:00Z",
                "home_team": "C",
                "away_team": "D",
                "competition": "EPL",
                "source_available_at_utc": "2026-09-15T19:30:00Z",
                "pit_verified": True,
                "starter_status": "ANNOUNCED",
            },
            {
                "match_id": "no-starters",
                "kickoff_utc": "2026-09-15T20:00:00Z",
                "home_team": "E",
                "away_team": "F",
                "competition": "EPL",
                "source_available_at_utc": "2026-09-14T00:00:00Z",
                "pit_verified": True,
                "starter_status": "EXPECTED",
            },
            {
                "match_id": "unverified",
                "kickoff_utc": "2026-09-15T21:00:00Z",
                "home_team": "G",
                "away_team": "H",
                "competition": "EPL",
                "source_available_at_utc": "2026-09-14T00:00:00Z",
                "pit_verified": False,
                "starter_status": "CONFIRMED",
            },
        ]
    )


def test_eligibility_is_forward_looking_without_requiring_official_lineups():
    d = _fixtures(Path("."))
    out = _eligible_fixtures(d, pd.Timestamp("2026-09-14T12:00:00Z"))
    assert out["match_id"].tolist() == ["ok", "no-starters"]


def test_missing_required_fixture_field_fails_closed():
    d = _fixtures(Path(".")).drop(columns=["source_available_at_utc"])
    with pytest.raises(RuntimeError, match="missing required columns"):
        _eligible_fixtures(d, pd.Timestamp("2026-09-14T12:00:00Z"))


def test_default_model_policy_is_production_fail_closed():
    import inspect
    from src.prediction.runner import run

    assert inspect.signature(run).parameters["model_policy"].default == "production"


def test_explicit_best_available_is_an_opt_in_resolution_path(tmp_path, monkeypatch):
    from src.prediction import runner

    calls = []

    def resolve():
        calls.append("resolve")
        return str(tmp_path / "candidate.pkl"), str(tmp_path / "candidate.json"), "VALIDATED_CANDIDATE"

    monkeypatch.setattr(runner, "resolve_best_available_paths", resolve)
    monkeypatch.setattr(
        runner,
        "load_best_available_model",
        lambda path: calls.append(("registry", path)) or {
            "adoption_status": "VALIDATED_CANDIDATE",
            "model_version": "candidate-v1",
        },
    )
    monkeypatch.setattr(
        runner,
        "load_bundle",
        lambda path: {"schema_version": 1, "model_version": "candidate-v1"},
    )

    status = runner.run(
        fixtures_path=str(tmp_path / "missing-fixtures.csv"),
        bundle_path="artifacts/production_model.pkl",
        registry_path="artifacts/model_registry.json",
        status_path=str(tmp_path / "status.json"),
        prediction_time="2026-09-15T12:00:00Z",
        model_policy="best_available",
    )

    assert calls == [
        "resolve",
        ("registry", str(tmp_path / "candidate.json")),
    ]
    assert status["status"] == "NO_FIXTURE_INPUT"



def test_prediction_output_carries_pit_provenance_metadata(tmp_path, monkeypatch):
    from src.prediction import runner

    monkeypatch.setattr(
        runner,
        "load_adopted_model",
        lambda path: {"adoption_status": "ADOPT", "model_version": "v1", "oos_verified": True},
    )
    monkeypatch.setattr(runner, "load_bundle", lambda path: {"model_version": "v1"})
    monkeypatch.setattr(runner, "predict_bundle", lambda bundle, X: [[0.60, 0.25, 0.15]])

    fixtures_path = tmp_path / "future.csv"
    output_path = tmp_path / "predictions.csv"
    status_path = tmp_path / "status.json"
    frame = pd.DataFrame([{
        "match_id": "m1",
        "kickoff_utc": "2026-09-15T18:00:00Z",
        "home_team": "A", "away_team": "B", "competition": "EPL",
        "source_available_at_utc": "2026-09-15T10:00:00Z",
        "pit_verified": True,
        "starter_status": "ANNOUNCED",
    }])
    frame.to_csv(fixtures_path, index=False)

    runner.run(
        fixtures_path=str(fixtures_path),
        bundle_path=str(tmp_path / "model.pkl"),
        output_path=str(output_path),
        status_path=str(status_path),
        prediction_time="2026-09-15T11:00:00Z",
        registry_path=str(tmp_path / "registry.json"),
    )

    result = pd.read_csv(output_path)
    assert {"source_available_at_utc", "pit_verified"} <= set(result.columns)
    assert bool(result.loc[0, "pit_verified"])
    assert result.loc[0, "source_available_at_utc"] == "2026-09-15 10:00:00+00:00"
