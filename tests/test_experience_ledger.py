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
