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
    predict,
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



def test_espn_event_adapter_preserves_current_match_state():
    from src.prediction.live_research_forecast import _espn_event_adapter, find_target

    raw = {
        "id": "98765",
        "date": "2026-10-08T18:45:00Z",
        "competitions": [{
            "competitors": [
                {"homeAway": "home", "team": {"displayName": "France"}, "score": "2"},
                {"homeAway": "away", "team": {"displayName": "Belgium"}, "score": "1"},
            ]
        }],
        "status": {"type": {"state": "in", "name": "STATUS_IN_PROGRESS"}},
    }
    event = _espn_event_adapter(raw, "uefa.nations")
    assert event is not None
    assert event["id"] == "espn:uefa.nations:98765"
    assert event["homeTeam"]["name"] == "France"
    assert event["awayTeam"]["name"] == "Belgium"
    assert event["homeScore"]["current"] == 2
    assert event["awayScore"]["current"] == 1
    assert event["status"]["type"] == "inprogress"
    assert find_target(event, load_config()[0]) == "France-Belgium"


def test_collect_events_uses_espn_fallback_when_sofascore_live_fails(monkeypatch):
    import src.prediction.live_research_forecast as module

    config, _ = load_config()
    config["runtime"]["schedule_days"] = 1
    calls = []

    espn_payload = {
        "events": [{
            "id": "24680",
            "date": "2026-10-08T18:45:00Z",
            "competitions": [{
                "competitors": [
                    {"homeAway": "home", "team": {"displayName": "Italy"}, "score": "0"},
                    {"homeAway": "away", "team": {"displayName": "Turkey"}, "score": "0"},
                ]
            }],
            "status": {"type": {"state": "pre", "name": "STATUS_SCHEDULED"}},
        }]
    }

    def fake_get_json(url, timeout):
        calls.append(url)
        if "sofascore.com/api/v1" in url:
            raise RuntimeError("403 Forbidden")
        assert "site.api.espn.com/apis/site/v2/sports/soccer/" in url
        return espn_payload, "2026-10-08T17:00:00Z", "digest"

    monkeypatch.setattr(module, "_get_json", fake_get_json)
    prediction_time = pd.Timestamp("2026-10-08T17:00:00Z")
    events, source_times, source_hashes = module.collect_events(prediction_time, config)

    assert len(events) == 1
    assert events[0]["id"].startswith("espn:")
    assert events[0]["homeTeam"]["name"] == "Italy"
    assert events[0]["awayTeam"]["name"] == "Turkey"
    assert any("espn_uefa.nations_" in key for key in source_times)
    assert "digest" in source_hashes.values()
    assert calls



def test_live_provenance_does_not_promote_retrieval_to_availability():
    config, config_hash = load_config()
    event = {
        "id": "123",
        "startTimestamp": int(pd.Timestamp("2026-10-08T18:45:00Z").timestamp()),
        "homeTeam": {"name": "France"},
        "awayTeam": {"name": "Belgium"},
        "status": {"type": "scheduled"},
        "homeScore": {"current": 0},
        "awayScore": {"current": 0},
        "tournament": {"uniqueTournament": {"name": "uefa.nations"}},
    }
    row = predict(
        event,
        {"france": 2070.0, "belgium": 1947.0},
        pd.Timestamp("2026-10-08T17:00:00Z"),
        {"espn_uefa.nations_20261008": "2026-10-08T17:00:00Z"},
        {"espn_uefa.nations_20261008": "digest"},
        config,
        config_hash,
        "elo-digest",
        event_source="espn",
    )
    assert row["source_available_at_utc"] is None
    assert row["source_available_lower_bound_utc"] is None
    assert row["source_retrieved_at_utc"] == "2026-10-08T17:00:00Z"
    assert row["event_source"] == "espn"
