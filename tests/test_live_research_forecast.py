import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.prediction.live_research_forecast import (
    REQUIRED,
    load_config,
    pre_match_lambdas,
    parse_elo_tsv,
    result_probs,
    score_matrix,
    verify,
)


def test_config_loads_and_is_hashed():
    config, digest = load_config()
    assert config["schema_version"] == 1
    assert len(digest) == 64
    assert set(config["target_teams"]) == {"france", "belgium", "italy", "turkey"}


def test_parse_world_elo_shape():
    text = (
        "1\tFrance\tFR\t2070\n"
        "2\tBelgium\tBE\t1947\n"
        "3\tItaly\tIT\t1869\n"
        "4\tTurkey\tTR\t1852\n"
    )
    ratings = parse_elo_tsv(
        text,
        {"FR": "france", "BE": "belgium", "IT": "italy", "TR": "turkey"},
    )
    assert ratings == {
        "france": 2070.0,
        "belgium": 1947.0,
        "italy": 1869.0,
        "turkey": 1852.0,
    }


def test_probability_contract():
    config, _ = load_config()
    home, away = pre_match_lambdas(2070.0, 1947.0, config)
    assert home > away > 0
    matrix = score_matrix(home, away, int(config["model"]["max_goals"]))
    probs = result_probs(matrix, float(config["model"]["probability_shrink"]))
    assert np.isfinite(probs).all()
    assert np.isclose(probs.sum(), 1.0)
    assert np.all((probs >= 0) & (probs <= 1))


def test_empty_output_contract_is_self_describing(tmp_path):
    path = tmp_path / "empty.csv"
    pd.DataFrame(columns=sorted(REQUIRED)).to_csv(path, index=False)
    result = verify(str(path))
    assert result["status"] == "VERIFIED"
    assert result["rows"] == 0
    assert result["empty_target_set"] is True


def test_verify_rejects_production_status(tmp_path):
    path = tmp_path / "forecast.csv"
    row = {
        "match_id": "x",
        "prediction_revision_id": "rev",
        "kickoff_utc": "2030-01-01T00:00:00Z",
        "home_team": "France",
        "away_team": "Belgium",
        "result_prediction": "Home",
        "p_home": 0.60,
        "p_draw": 0.20,
        "p_away": 0.20,
        "score_1": "1-0",
        "score_1_probability": 0.20,
        "score_2": "2-0",
        "score_2_probability": 0.15,
        "score_3": "1-1",
        "score_3_probability": 0.10,
        "prediction_state": "PREMATCH_RESEARCH_FORECAST",
        "pit_status": "CURRENT_OBSERVED_PRE_KICKOFF",
        "production_status": "PRODUCTION",
        "data_completeness": 1.0,
        "uncertainty": 0.4,
        "predictability": 0.6,
        "source_snapshot_hash": "a",
        "source_snapshot_hashes": "{}",
        "config_sha256": "b",
        "git_commit_sha": "c",
        "experiment_fingerprint": "d",
    }
    pd.DataFrame([row]).to_csv(path, index=False)
    try:
        verify(str(path))
    except RuntimeError as exc:
        assert "production flag" in str(exc)
    else:
        raise AssertionError("Production must be rejected by the research contract")


def test_verify_accepts_research_contract(tmp_path):
    path = tmp_path / "forecast.csv"
    row = {
        "match_id": "x",
        "prediction_revision_id": "rev",
        "kickoff_utc": "2030-01-01T00:00:00Z",
        "home_team": "France",
        "away_team": "Belgium",
        "result_prediction": "Home",
        "p_home": 0.60,
        "p_draw": 0.20,
        "p_away": 0.20,
        "score_1": "1-0",
        "score_1_probability": 0.20,
        "score_2": "2-0",
        "score_2_probability": 0.15,
        "score_3": "1-1",
        "score_3_probability": 0.10,
        "prediction_state": "PREMATCH_RESEARCH_FORECAST",
        "pit_status": "CURRENT_OBSERVED_PRE_KICKOFF",
        "production_status": "NOT_PRODUCTION",
        "data_completeness": 1.0,
        "uncertainty": 0.4,
        "predictability": 0.6,
        "source_snapshot_hash": "a",
        "source_snapshot_hashes": "{}",
        "config_sha256": "b",
        "git_commit_sha": "c",
        "experiment_fingerprint": "d",
    }
    pd.DataFrame([row]).to_csv(path, index=False)
    assert verify(str(path))["status"] == "VERIFIED"
