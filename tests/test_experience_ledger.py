import pandas as pd
from scripts.experience_ledger import _key, record_prediction_file

def test_fixture_key_is_stable():
    assert _key("2026-09-26T10:00:00Z","Team A","Team B") == _key(
        "2026-09-26T10:00:00+00:00"," Team A ","Team B"
    )

def test_record_deduplicates_prediction_state(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.csv"
    predictions = tmp_path / "predictions.csv"
    monkeypatch.setattr("scripts.experience_ledger.LEDGER", ledger)
    row = {
        "match_id":"espn:1","kickoff_utc":"2026-09-26T10:00:00Z","home_team":"A","away_team":"B",
        "competition":"EPL","p_home":0.5,"p_draw":0.25,"p_away":0.25,"model_version":"v1",
        "score_1":"1-0","score_1_probability":0.2,"score_2":"0-0","score_2_probability":0.1,
        "score_3":"1-1","score_3_probability":0.1,
    }
    pd.DataFrame([row]).to_csv(predictions,index=False)
    assert record_prediction_file(str(predictions))["added"] == 1
    assert record_prediction_file(str(predictions))["added"] == 0
    assert len(pd.read_csv(ledger)) == 1


def test_record_rejects_invalid_1x2_probabilities(tmp_path, monkeypatch):
    from scripts import experience_ledger as mod
    import pytest

    ledger = tmp_path / "ledger.csv"
    predictions = tmp_path / "predictions.csv"
    monkeypatch.setattr(mod, "LEDGER", ledger)
    pd.DataFrame([{
        "match_id": "espn:bad",
        "kickoff_utc": "2026-09-26T10:00:00Z",
        "home_team": "A",
        "away_team": "B",
        "competition": "EPL",
        "p_home": 0.8,
        "p_draw": 0.8,
        "p_away": -0.6,
        "model_version": "v1",
    }]).to_csv(predictions, index=False)
    with pytest.raises(RuntimeError, match="1X2 probabilities"):
        mod.record_prediction_file(str(predictions))


def test_compute_metrics_includes_proper_probability_scores(tmp_path, monkeypatch):
    from scripts import experience_ledger as mod

    metrics_path = tmp_path / "metrics.csv"
    status_path = tmp_path / "status.json"
    monkeypatch.setattr(mod, "METRICS", metrics_path)
    monkeypatch.setattr(mod, "STATUS", status_path)
    ledger = pd.DataFrame([
        {"kickoff_utc":"2026-09-01T10:00:00Z","actual_result":"H","p_home":0.8,"p_draw":0.1,"p_away":0.1,
         "correct_1x2":1,"score_top1_hit":0,"score_top3_hit":1,"model_version":"v1","competition":"EPL"},
        {"kickoff_utc":"2026-09-02T10:00:00Z","actual_result":"D","p_home":0.1,"p_draw":0.8,"p_away":0.1,
         "correct_1x2":1,"score_top1_hit":0,"score_top3_hit":1,"model_version":"v1","competition":"EPL"},
        {"kickoff_utc":"2026-09-03T10:00:00Z","actual_result":"A","p_home":0.1,"p_draw":0.1,"p_away":0.8,
         "correct_1x2":1,"score_top1_hit":0,"score_top3_hit":1,"model_version":"v1","competition":"EPL"},
    ])
    assert mod.compute_metrics(ledger) > 0
    report = pd.read_csv(metrics_path)
    row = report.loc[report["scope"] == "all"].iloc[0]
    assert float(row["logloss"]) > 0
    assert float(row["brier"]) >= 0
    assert float(row["rps"]) >= 0
    assert float(row["ece"]) >= 0
    assert int(row["probability_rows"]) == 3


def test_record_normalizes_tiny_probability_rounding(tmp_path, monkeypatch):
    from scripts import experience_ledger as mod
    ledger = tmp_path / "ledger.csv"
    predictions = tmp_path / "predictions.csv"
    monkeypatch.setattr(mod, "LEDGER", ledger)
    pd.DataFrame([{
        "match_id": "espn:rounding",
        "kickoff_utc": "2026-09-26T11:00:00Z",
        "home_team": "A",
        "away_team": "B",
        "competition": "EPL",
        "p_home": 0.33333,
        "p_draw": 0.33333,
        "p_away": 0.33334,
        "model_version": "v1",
    }]).to_csv(predictions, index=False)
    result = mod.record_prediction_file(str(predictions))
    assert result["added"] == 1
    row = pd.read_csv(ledger).iloc[0]
    assert float(row["p_home"]) + float(row["p_draw"]) + float(row["p_away"]) == pytest.approx(1.0, abs=1e-5)
