import json
import pytest
import pandas as pd
from scripts.experience_ledger import _key, record_prediction_file

def test_fixture_key_is_stable():
    assert _key("2026-09-26T10:00:00Z","Team A","Team B") == _key(
        "2026-09-26T10:00:00+00:00"," Team A ","Team B"
    )

def test_record_deduplicates_prediction_state(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.csv"
    snapshots = tmp_path / "prediction_snapshots.jsonl"
    predictions = tmp_path / "predictions.csv"
    monkeypatch.setattr("scripts.experience_ledger.LEDGER", ledger)
    monkeypatch.setattr("scripts.experience_ledger.PREDICTION_SNAPSHOTS", snapshots)
    row = {
        "match_id":"espn:1","kickoff_utc":"2026-09-26T10:00:00Z","prediction_time_utc":"2026-09-26T08:00:00Z","source_available_at_utc":"2026-09-26T07:30:00Z","pit_verified":True,"home_team":"A","away_team":"B",
        "competition":"EPL","p_home":0.5,"p_draw":0.25,"p_away":0.25,"model_version":"v1",
        "score_1":"1-0","score_1_probability":0.2,"score_2":"0-0","score_2_probability":0.1,
        "score_3":"1-1","score_3_probability":0.1,
    }
    pd.DataFrame([row]).to_csv(predictions,index=False)
    assert record_prediction_file(str(predictions))["added"] == 1
    assert record_prediction_file(str(predictions))["added"] == 0
    recorded = pd.read_csv(ledger)
    assert len(recorded) == 1
    assert "experience_available_at_utc" in recorded.columns
    assert pd.isna(recorded.iloc[0]["experience_available_at_utc"])
    snapshot_rows = [json.loads(x) for x in snapshots.read_text().splitlines() if x.strip()]
    assert len(snapshot_rows) == 1
    assert snapshot_rows[0]["prediction_pit_gate"] == "PASS"
    assert snapshot_rows[0]["prediction_pit_cutoff_utc"].startswith("2026-09-26T08:00:00")


def test_record_keeps_distinct_prediction_times_as_distinct_states(tmp_path, monkeypatch):
    from scripts import experience_ledger as mod

    ledger = tmp_path / "ledger.csv"
    snapshots = tmp_path / "prediction_snapshots.jsonl"
    predictions = tmp_path / "predictions.csv"
    monkeypatch.setattr(mod, "LEDGER", ledger)
    monkeypatch.setattr(mod, "PREDICTION_SNAPSHOTS", snapshots)

    base = {
        "match_id": "espn:2",
        "kickoff_utc": "2026-09-26T10:00:00Z",
        "home_team": "A",
        "away_team": "B",
        "competition": "EPL",
        "p_home": 0.5,
        "p_draw": 0.25,
        "p_away": 0.25,
        "model_version": "v1",
        "source_available_at_utc": "2026-09-26T07:30:00Z",
        "pit_verified": True,
        "score_1": "1-0",
        "score_1_probability": 0.2,
        "score_2": "0-0",
        "score_2_probability": 0.1,
        "score_3": "1-1",
        "score_3_probability": 0.1,
    }
    first = {**base, "prediction_time_utc": "2026-09-26T08:00:00Z"}
    second = {**base, "prediction_time_utc": "2026-09-26T09:00:00Z"}

    pd.DataFrame([first]).to_csv(predictions, index=False)
    assert mod.record_prediction_file(str(predictions))["added"] == 1
    pd.DataFrame([second]).to_csv(predictions, index=False)
    assert mod.record_prediction_file(str(predictions))["added"] == 1

    recorded = pd.read_csv(ledger)
    assert len(recorded) == 2
    assert set(recorded["prediction_pit_cutoff_utc"]) == {
        "2026-09-26T08:00:00+00:00",
        "2026-09-26T09:00:00+00:00",
    }
    snapshot_rows = [json.loads(x) for x in snapshots.read_text().splitlines() if x.strip()]
    assert len(snapshot_rows) == 2
    assert len({row["prediction_state_id"] for row in snapshot_rows}) == 2


def test_record_rejects_missing_source_availability(tmp_path, monkeypatch):
    from scripts import experience_ledger as mod

    predictions = tmp_path / "predictions.csv"
    monkeypatch.setattr(mod, "LEDGER", tmp_path / "ledger.csv")
    pd.DataFrame([{
        "match_id": "espn:no-source",
        "kickoff_utc": "2026-09-26T10:00:00Z",
        "prediction_time_utc": "2026-09-26T08:00:00Z",
        "pit_verified": True,
        "home_team": "A",
        "away_team": "B",
        "competition": "EPL",
        "p_home": 0.5,
        "p_draw": 0.25,
        "p_away": 0.25,
        "model_version": "v1",
    }]).to_csv(predictions, index=False)
    with pytest.raises(RuntimeError, match="source_available_at_utc or available_at_utc"):
        mod.record_prediction_file(str(predictions))


def test_record_rejects_missing_pit_verified(tmp_path, monkeypatch):
    from scripts import experience_ledger as mod

    predictions = tmp_path / "predictions.csv"
    monkeypatch.setattr(mod, "LEDGER", tmp_path / "ledger.csv")
    pd.DataFrame([{
        "match_id": "espn:no-pit-flag",
        "kickoff_utc": "2026-09-26T10:00:00Z",
        "prediction_time_utc": "2026-09-26T08:00:00Z",
        "source_available_at_utc": "2026-09-26T07:30:00Z",
        "home_team": "A",
        "away_team": "B",
        "competition": "EPL",
        "p_home": 0.5,
        "p_draw": 0.25,
        "p_away": 0.25,
        "model_version": "v1",
    }]).to_csv(predictions, index=False)
    with pytest.raises(RuntimeError, match="pit_verified"):
        mod.record_prediction_file(str(predictions))


def test_record_rejects_unverified_pit_state(tmp_path, monkeypatch):
    from scripts import experience_ledger as mod

    predictions = tmp_path / "predictions.csv"
    monkeypatch.setattr(mod, "LEDGER", tmp_path / "ledger.csv")
    pd.DataFrame([{
        "match_id": "espn:unverified",
        "kickoff_utc": "2026-09-26T10:00:00Z",
        "prediction_time_utc": "2026-09-26T08:00:00Z",
        "source_available_at_utc": "2026-09-26T07:30:00Z",
        "pit_verified": False,
        "home_team": "A",
        "away_team": "B",
        "competition": "EPL",
        "p_home": 0.5,
        "p_draw": 0.25,
        "p_away": 0.25,
        "model_version": "v1",
    }]).to_csv(predictions, index=False)
    with pytest.raises(RuntimeError, match="pit_verified=false"):
        mod.record_prediction_file(str(predictions))


def test_record_rejects_source_available_after_cutoff(tmp_path, monkeypatch):
    from scripts import experience_ledger as mod

    predictions = tmp_path / "predictions.csv"
    monkeypatch.setattr(mod, "LEDGER", tmp_path / "ledger.csv")
    pd.DataFrame([{
        "match_id": "espn:late-source",
        "kickoff_utc": "2026-09-26T10:00:00Z",
        "prediction_time_utc": "2026-09-26T08:00:00Z",
        "source_available_at_utc": "2026-09-26T08:01:00Z",
        "pit_verified": True,
        "home_team": "A",
        "away_team": "B",
        "competition": "EPL",
        "p_home": 0.5,
        "p_draw": 0.25,
        "p_away": 0.25,
        "model_version": "v1",
    }]).to_csv(predictions, index=False)
    with pytest.raises(RuntimeError, match="after prediction cutoff"):
        mod.record_prediction_file(str(predictions))


def test_record_rejects_retrieval_after_cutoff(tmp_path, monkeypatch):
    from scripts import experience_ledger as mod

    predictions = tmp_path / "predictions.csv"
    monkeypatch.setattr(mod, "LEDGER", tmp_path / "ledger.csv")
    pd.DataFrame([{
        "match_id": "espn:late-retrieval",
        "kickoff_utc": "2026-09-26T10:00:00Z",
        "prediction_time_utc": "2026-09-26T08:00:00Z",
        "source_available_at_utc": "2026-09-26T07:30:00Z",
        "pit_verified": True,
        "retrieved_at_utc": "2026-09-26T08:01:00Z",
        "home_team": "A",
        "away_team": "B",
        "competition": "EPL",
        "p_home": 0.5,
        "p_draw": 0.25,
        "p_away": 0.25,
        "model_version": "v1",
    }]).to_csv(predictions, index=False)
    with pytest.raises(RuntimeError, match="retrieval occurs after prediction cutoff"):
        mod.record_prediction_file(str(predictions))


def test_record_normalizes_available_alias_and_validates_ordering(tmp_path, monkeypatch):
    from scripts import experience_ledger as mod

    ledger = tmp_path / "ledger.csv"
    predictions = tmp_path / "predictions.csv"
    monkeypatch.setattr(mod, "LEDGER", ledger)
    pd.DataFrame([{
        "match_id": "espn:alias",
        "kickoff_utc": "2026-09-26T10:00:00Z",
        "prediction_time_utc": "2026-09-26T08:00:00Z",
        "available_at_utc": "2026-09-26T07:30:00Z",
        "pit_verified": True,
        "published_at_utc": "2026-09-26T07:35:00Z",
        "retrieved_at_utc": "2026-09-26T07:40:00Z",
        "home_team": "A",
        "away_team": "B",
        "competition": "EPL",
        "p_home": 0.5,
        "p_draw": 0.25,
        "p_away": 0.25,
        "model_version": "v1",
    }]).to_csv(predictions, index=False)

    assert mod.record_prediction_file(str(predictions))["added"] == 1
    recorded = pd.read_csv(ledger)
    assert recorded.loc[0, "source_available_at_utc"] == "2026-09-26T07:30:00+00:00"
    assert recorded.loc[0, "available_at_utc"] == "2026-09-26T07:30:00+00:00"


def test_record_rejects_disagreeing_availability_aliases(tmp_path, monkeypatch):
    from scripts import experience_ledger as mod

    predictions = tmp_path / "predictions.csv"
    monkeypatch.setattr(mod, "LEDGER", tmp_path / "ledger.csv")
    pd.DataFrame([{
        "match_id": "espn:alias-conflict",
        "kickoff_utc": "2026-09-26T10:00:00Z",
        "prediction_time_utc": "2026-09-26T08:00:00Z",
        "source_available_at_utc": "2026-09-26T07:30:00Z",
        "available_at_utc": "2026-09-26T07:31:00Z",
        "pit_verified": True,
        "home_team": "A",
        "away_team": "B",
        "competition": "EPL",
        "p_home": 0.5,
        "p_draw": 0.25,
        "p_away": 0.25,
        "model_version": "v1",
    }]).to_csv(predictions, index=False)
    with pytest.raises(RuntimeError, match="aliases disagree"):
        mod.record_prediction_file(str(predictions))


def test_record_allows_publication_before_source_availability(tmp_path, monkeypatch):
    from scripts import experience_ledger as mod

    predictions = tmp_path / "predictions.csv"
    monkeypatch.setattr(mod, "LEDGER", tmp_path / "ledger.csv")
    pd.DataFrame([{
        "match_id": "espn:pub-before-source",
        "kickoff_utc": "2026-09-26T10:00:00Z",
        "prediction_time_utc": "2026-09-26T08:00:00Z",
        "source_available_at_utc": "2026-09-26T07:30:00Z",
        "pit_verified": True,
        "published_at_utc": "2026-09-26T07:00:00Z",
        "retrieved_at_utc": "2026-09-26T07:40:00Z",
        "home_team": "A",
        "away_team": "B",
        "competition": "EPL",
        "p_home": 0.5,
        "p_draw": 0.25,
        "p_away": 0.25,
        "model_version": "v1",
    }]).to_csv(predictions, index=False)
    assert mod.record_prediction_file(str(predictions))["added"] == 1


def test_record_rejects_invalid_1x2_probabilities(tmp_path, monkeypatch):
    from scripts import experience_ledger as mod
    import pytest

    ledger = tmp_path / "ledger.csv"
    predictions = tmp_path / "predictions.csv"
    monkeypatch.setattr(mod, "LEDGER", ledger)
    pd.DataFrame([{
        "match_id": "espn:bad",
        "kickoff_utc": "2026-09-26T10:00:00Z",
        "prediction_time_utc": "2026-09-26T08:00:00Z",
        "source_available_at_utc": "2026-09-26T07:30:00Z",
        "pit_verified": True,
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


def test_compute_metrics_status_contains_monitoring_summary(tmp_path, monkeypatch):
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
    ])
    mod.compute_metrics(ledger)
    import json
    status = json.loads(status_path.read_text())
    assert "summary" in status
    assert status["summary"]["all"]["n"] == 2
    assert "logloss" in status["summary"]["all"]


def test_compute_metrics_rejects_invalid_actual_result_label(tmp_path, monkeypatch):
    from scripts import experience_ledger as mod

    monkeypatch.setattr(mod, "METRICS", tmp_path / "metrics.csv")
    monkeypatch.setattr(mod, "STATUS", tmp_path / "status.json")
    ledger = pd.DataFrame([
        {"kickoff_utc":"2026-09-01T10:00:00Z","actual_result":"HOME","p_home":0.8,"p_draw":0.1,"p_away":0.1,
         "correct_1x2":1,"score_top1_hit":0,"score_top3_hit":1,"model_version":"v1","competition":"EPL"},
    ])
    with pytest.raises(RuntimeError, match="invalid actual_result labels"):
        mod.compute_metrics(ledger)


def test_compute_metrics_rejects_invalid_probability_row(tmp_path, monkeypatch):
    from scripts import experience_ledger as mod

    monkeypatch.setattr(mod, "METRICS", tmp_path / "metrics.csv")
    monkeypatch.setattr(mod, "STATUS", tmp_path / "status.json")
    ledger = pd.DataFrame([
        {"kickoff_utc":"2026-09-01T10:00:00Z","actual_result":"H","p_home":1.2,"p_draw":-0.1,"p_away":0.0,
         "correct_1x2":1,"score_top1_hit":0,"score_top3_hit":1,"model_version":"v1","competition":"EPL"},
    ])
    with pytest.raises(RuntimeError, match="invalid 1X2 probability rows"):
        mod.compute_metrics(ledger)


def test_metric_deltas_compare_previous_status(tmp_path, monkeypatch):
    from scripts import experience_ledger as mod
    monkeypatch.setattr(mod, "STATUS", tmp_path / "status.json")
    previous = {"summary": {"all": {"1x2_accuracy_pct": 50.0, "logloss": 1.0, "brier": 0.5, "rps": 0.3, "ece": 0.1}}}
    current = {"summary": {"all": {"1x2_accuracy_pct": 52.0, "logloss": 0.9, "brier": 0.5, "rps": 0.25, "ece": 0.08}}}
    assert mod._status_metric_deltas(previous, current) == {
        "all": {"1x2_accuracy_pct": 2.0, "logloss": -0.1, "rps": -0.05, "ece": -0.02}
    }


def test_compute_metrics_persists_metric_deltas(tmp_path, monkeypatch):
    from scripts import experience_ledger as mod
    metrics_path = tmp_path / "metrics.csv"
    status_path = tmp_path / "status.json"
    monkeypatch.setattr(mod, "METRICS", metrics_path)
    monkeypatch.setattr(mod, "STATUS", status_path)
    status_path.write_text(
        '{"status":"OK","summary":{"all":{"1x2_accuracy_pct":50.0,"logloss":1.0,"brier":0.5,"rps":0.3,"ece":0.1}}}',
        encoding="utf-8",
    )
    ledger = pd.DataFrame([
        {"kickoff_utc":"2026-09-01T10:00:00Z","actual_result":"H","p_home":0.8,"p_draw":0.1,"p_away":0.1,
         "correct_1x2":1,"score_top1_hit":0,"score_top3_hit":1,"model_version":"v1","competition":"EPL"},
        {"kickoff_utc":"2026-09-02T10:00:00Z","actual_result":"D","p_home":0.1,"p_draw":0.8,"p_away":0.1,
         "correct_1x2":1,"score_top1_hit":0,"score_top3_hit":1,"model_version":"v1","competition":"EPL"},
    ])
    mod.compute_metrics(ledger)
    status = json.loads(status_path.read_text())
    assert status["metric_deltas_vs_previous"]["all"]["1x2_accuracy_pct"] == 50.0

def test_compute_metrics_publishes_zero_state_status(tmp_path, monkeypatch):
    from scripts import experience_ledger as mod

    metrics_path = tmp_path / "metrics.csv"
    status_path = tmp_path / "status.json"
    monkeypatch.setattr(mod, "METRICS", metrics_path)
    monkeypatch.setattr(mod, "STATUS", status_path)

    mod.compute_metrics(pd.DataFrame())
    status = json.loads(status_path.read_text())
    assert status["status"] == "NO_SETTLED_PREDICTIONS"
    assert status["summary"]["all"]["n"] == 0
    assert status["summary"]["all"]["probability_rows"] == 0


def test_status_command_publishes_zero_state(tmp_path, monkeypatch):
    from scripts import experience_ledger as mod
    import sys

    monkeypatch.setattr(mod, "LEDGER", tmp_path / "ledger.csv")
    monkeypatch.setattr(mod, "METRICS", tmp_path / "metrics.csv")
    monkeypatch.setattr(mod, "STATUS", tmp_path / "status.json")
    monkeypatch.setattr(sys, "argv", ["experience_ledger", "status"])

    assert mod.main() is None
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "NO_SETTLED_PREDICTIONS"
    assert status["summary"]["all"]["n"] == 0


def test_preserve_timestamp_when_experience_state_is_unchanged():
    from scripts.experience_ledger import _preserve_timestamp_when_unchanged

    previous = {
        "status": "NO_SETTLED_PREDICTIONS",
        "settled_predictions": 0,
        "ledger_rows": 0,
        "generated_at_utc": "2026-09-26T00:00:00+00:00",
        "summary": {"all": {"n": 0, "probability_rows": 0}},
        "metric_deltas_vs_previous": {},
    }
    current = dict(previous)
    current["generated_at_utc"] = "2026-09-26T04:00:00+00:00"
    assert _preserve_timestamp_when_unchanged(previous, current)["generated_at_utc"] == previous["generated_at_utc"]


def test_update_timestamp_when_experience_state_changes():
    from scripts.experience_ledger import _preserve_timestamp_when_unchanged

    previous = {
        "status": "NO_SETTLED_PREDICTIONS",
        "settled_predictions": 0,
        "ledger_rows": 0,
        "generated_at_utc": "2026-09-26T00:00:00+00:00",
        "summary": {"all": {"n": 0, "probability_rows": 0}},
    }
    current = dict(previous)
    current["ledger_rows"] = 1
    current["generated_at_utc"] = "2026-09-26T04:00:00+00:00"
    assert _preserve_timestamp_when_unchanged(previous, current)["generated_at_utc"] == current["generated_at_utc"]
def test_compute_metrics_persists_target_specific_results(tmp_path, monkeypatch):
    from scripts import experience_ledger as mod

    metrics_path = tmp_path / "metrics.csv"
    target_metrics_path = tmp_path / "target_metrics.csv"
    status_path = tmp_path / "status.json"
    monkeypatch.setattr(mod, "METRICS", metrics_path)
    monkeypatch.setattr(mod, "TARGET_METRICS", target_metrics_path)
    monkeypatch.setattr(mod, "STATUS", status_path)

    base = {
        "kickoff_utc": "2026-09-01T10:00:00Z",
        "actual_result": "H",
        "p_home": 0.8,
        "p_draw": 0.1,
        "p_away": 0.1,
        "correct_1x2": 1,
        "score_top1_hit": 0,
        "score_top3_hit": 1,
        "over_2_5_correct": 1,
        "btts_correct": 0,
        "mom_top1_hit": 0,
        "mom_top4_hit": 1,
        "model_version": "v1",
        "competition": "EPL",
    }
    pd.DataFrame([base]).to_csv(tmp_path / "ledger.csv", index=False)

    assert mod.compute_metrics(pd.DataFrame([base])) > 0
    report = pd.read_csv(target_metrics_path)
    assert set(report["target"]) == {"1X2", "Score", "O/U", "BTTS", "MOM"}
    assert report.loc[(report["target"] == "1X2") & (report["metric"] == "1x2_accuracy_pct"), "accuracy_pct"].iloc[0] == 100.0
    assert report.loc[(report["target"] == "Score") & (report["metric"] == "score_top3_accuracy_pct"), "accuracy_pct"].iloc[0] == 100.0
    assert report.loc[(report["target"] == "MOM") & (report["metric"] == "mom_top4_accuracy_pct"), "accuracy_pct"].iloc[0] == 100.0
def test_record_normalizes_binary_target_probabilities(tmp_path, monkeypatch):
    from scripts import experience_ledger as mod

    ledger = tmp_path / "ledger.csv"
    predictions = tmp_path / "predictions.csv"
    monkeypatch.setattr(mod, "LEDGER", ledger)
    monkeypatch.setattr(mod, "PREDICTION_SNAPSHOTS", tmp_path / "snapshots.jsonl")

    row = {
        "match_id": "espn:3",
        "kickoff_utc": "2026-09-26T10:00:00Z",
        "prediction_time_utc": "2026-09-26T08:00:00Z",
        "source_available_at_utc": "2026-09-26T07:30:00Z",
        "pit_verified": True,
        "home_team": "A",
        "away_team": "B",
        "competition": "EPL",
        "p_home": 0.5,
        "p_draw": 0.25,
        "p_away": 0.25,
        "market_over_2_5": 0.6,
        "market_under_2_5": 0.4,
        "market_btts_yes": 0.7,
        "market_btts_no": 0.3,
        "model_version": "v1",
    }
    pd.DataFrame([row]).to_csv(predictions, index=False)
    assert mod.record_prediction_file(str(predictions))["added"] == 1
    saved = pd.read_csv(ledger).iloc[0]
    assert float(saved["over_2_5"]) == 0.6
    assert float(saved["under_2_5"]) == 0.4
    assert float(saved["btts_yes"]) == 0.7
    assert float(saved["btts_no"]) == 0.3


def test_target_metrics_include_binary_probability_quality(tmp_path, monkeypatch):
    from scripts import experience_ledger as mod

    target_metrics_path = tmp_path / "target_metrics.csv"
    monkeypatch.setattr(mod, "METRICS", tmp_path / "metrics.csv")
    monkeypatch.setattr(mod, "TARGET_METRICS", target_metrics_path)
    monkeypatch.setattr(mod, "STATUS", tmp_path / "status.json")

    ledger = pd.DataFrame([{
        "kickoff_utc": "2026-09-01T10:00:00Z",
        "actual_result": "H",
        "actual_home_goals": 2,
        "actual_away_goals": 1,
        "p_home": 0.8,
        "p_draw": 0.1,
        "p_away": 0.1,
        "correct_1x2": 1,
        "score_top1_hit": 1,
        "score_top3_hit": 1,
        "over_2_5": 0.75,
        "under_2_5": 0.25,
        "over_2_5_correct": 1,
        "btts_yes": 0.8,
        "btts_no": 0.2,
        "btts_correct": 1,
        "model_version": "v1",
        "competition": "EPL",
    }])
    assert mod.compute_metrics(ledger) > 0
    report = pd.read_csv(target_metrics_path)
    ou = report[(report["scope"] == "all") & (report["target"] == "O/U")].iloc[0]
    btts = report[(report["scope"] == "all") & (report["target"] == "BTTS")].iloc[0]
    assert float(ou["accuracy_pct"]) == 100.0
    assert float(ou["logloss"]) > 0.0
    assert float(ou["brier"]) >= 0.0
    assert float(ou["ece"]) >= 0.0
    assert float(btts["accuracy_pct"]) == 100.0
    assert float(btts["logloss"]) > 0.0
    assert float(btts["brier"]) >= 0.0
    assert float(btts["ece"]) >= 0.0




def test_record_prediction_joins_outcome_free_shadow_telemetry(tmp_path, monkeypatch):
    from scripts import experience_ledger as mod

    ledger = tmp_path / "ledger.csv"
    snapshots = tmp_path / "snapshots.jsonl"
    predictions = tmp_path / "predictions.csv"
    diagnostics = tmp_path / "shadow.csv"
    monkeypatch.setattr(mod, "LEDGER", ledger)
    monkeypatch.setattr(mod, "PREDICTION_SNAPSHOTS", snapshots)

    pd.DataFrame([{
        "match_id": "m1",
        "kickoff_utc": "2026-09-26T10:00:00Z",
        "prediction_time_utc": "2026-09-26T08:00:00Z",
        "source_available_at_utc": "2026-09-26T07:30:00Z",
        "pit_verified": True,
        "home_team": "A",
        "away_team": "B",
        "competition": "EPL",
        "p_home": 0.6,
        "p_draw": 0.2,
        "p_away": 0.2,
        "model_version": "v1",
    }]).to_csv(predictions, index=False)
    pd.DataFrame([{
        "match_id": "m1",
        "model_disagreement": 0.2,
        "predictive_entropy": 0.4,
        "uncertainty_score": 0.3,
        "covariate_drift": 0.1,
        "history_support_risk": 0.5,
        "routing_risk": 0.35,
        "routing_risk_bucket": "MEDIUM",
        "routing_route": "GLOBAL",
    }]).to_csv(diagnostics, index=False)

    result = mod.record_prediction_file(str(predictions), diagnostics_path=str(diagnostics))
    assert result["shadow_telemetry_joined"] == 1
    assert result["shadow_telemetry_coverage"] == "FULL"
    saved = pd.read_csv(ledger).iloc[0]
    assert float(saved["shadow_uncertainty_score"]) == 0.3
    assert saved["shadow_routing_risk_bucket"] == "MEDIUM"


def _settlement_test_row():
    return pd.Series({
        "match_id": "test-match",
        "fixture_key": "2026-10-01T18:00:00+00:00|teama|teamb",
        "kickoff_utc": "2026-10-01T18:00:00Z",
        "prediction_pit_cutoff_utc": "2026-10-01T16:00:00Z",
        "prediction_pit_gate": "PASS",
        "p_home": 0.60,
        "p_draw": 0.20,
        "p_away": 0.20,
        "score_1": "1-0",
        "score_2": "1-1",
        "score_3": "2-0",
        "market_over_2_5": 0.50,
        "market_btts_yes": 0.50,
    })


def _sofa_event(hg=2, ag=1):
    return {
        "id": 101,
        "startTimestamp": 1790877600,
        "homeTeam": {"name": "Team A"},
        "awayTeam": {"name": "Team B"},
        "homeScore": {"current": hg},
        "awayScore": {"current": ag},
        "status": {"type": "finished"},
    }


def _espn_event(hg=2, ag=1):
    return {
        "id": "espn-101",
        "date": "2026-10-01T18:00:00Z",
        "competitions": [{
            "competitors": [
                {"homeAway": "home", "team": {"displayName": "Team A"}, "score": str(hg)},
                {"homeAway": "away", "team": {"displayName": "Team B"}, "score": str(ag)},
            ],
            "status": {"type": {"name": "STATUS_FINAL"}},
        }],
    }


def test_settlement_requires_matching_outcomes_when_both_sources_are_finished():
    from scripts.experience_ledger import _settle_row
    row = _settlement_test_row()
    out = _settle_row(
        row,
        {"2026-10-01T18:00:00+00:00|teama|teamb": _sofa_event(2, 1)},
        {},
        {"2026-10-01T18:00:00+00:00|teama|teamb": _espn_event(2, 1)},
        {},
    )
    assert out["settlement_verification"] == "DUAL_SOURCE_AGREE"
    assert out["settlement_sources_count"] == 2
    assert out["actual_score"] == "2-1"


def test_settlement_does_not_teach_from_source_disagreement():
    from scripts.experience_ledger import _settle_row
    row = _settlement_test_row()
    out = _settle_row(
        row,
        {"2026-10-01T18:00:00+00:00|teama|teamb": _sofa_event(2, 1)},
        {},
        {"2026-10-01T18:00:00+00:00|teama|teamb": _espn_event(1, 0)},
        {},
    )
    assert out["settlement_verification"] == "SOURCE_DISAGREEMENT"
    assert "actual_result" not in out
    assert "correct_1x2" not in out


def test_settlement_records_single_source_without_marking_consensus():
    from scripts.experience_ledger import _settle_row
    row = _settlement_test_row()
    out = _settle_row(
        row,
        {},
        {},
        {},
        {"test-match": _espn_event(2, 1)},
    )
    assert out["settlement_verification"] == "SINGLE_SOURCE"
    assert out["settlement_sources_count"] == 1
    assert out["settlement_source"] == "espn"

def test_compute_metrics_emits_explicit_scope_breakdowns(tmp_path, monkeypatch):
    from scripts import experience_ledger as mod

    metrics_path = tmp_path / "metrics.csv"
    target_metrics_path = tmp_path / "target_metrics.csv"
    status_path = tmp_path / "status.json"
    monkeypatch.setattr(mod, "METRICS", metrics_path)
    monkeypatch.setattr(mod, "TARGET_METRICS", target_metrics_path)
    monkeypatch.setattr(mod, "STATUS", status_path)

    rows = []
    for match_id, comp, phase, tier, regime, result in [
        ("m1", "EPL", "REGULAR", "TIER1", "STABLE", "H"),
        ("m2", "UCL", "KNOCKOUT", "TIER1", "VOLATILE", "A"),
        ("m3", "EPL", "REGULAR", "TIER1", "STABLE", "D"),
    ]:
        rows.append({
            "match_id": match_id,
            "kickoff_utc": "2026-09-01T10:00:00Z",
            "actual_result": result,
            "p_home": 0.60,
            "p_draw": 0.20,
            "p_away": 0.20,
            "correct_1x2": int(result == "H"),
            "score_top1_hit": 0,
            "score_top3_hit": 1,
            "over_2_5_correct": 1,
            "btts_correct": 0,
            "mom_top1_hit": 0,
            "mom_top4_hit": 1,
            "model_version": "v1",
            "competition": comp,
            "season_start": 2025,
            "phase": phase,
            "tier": tier,
            "regime": regime,
        })

    ledger = pd.DataFrame(rows)
    assert mod.compute_metrics(ledger) > 0

    report = pd.read_csv(metrics_path)
    assert set(report.loc[report["scope"] == "competition", "segment"]) == {"EPL", "UCL"}
    assert set(report.loc[report["scope"] == "season", "segment"]) == {"2025"}
    assert set(report.loc[report["scope"] == "phase", "segment"]) == {"REGULAR", "KNOCKOUT"}
    assert set(report.loc[report["scope"] == "tier", "segment"]) == {"TIER1"}
    assert set(report.loc[report["scope"] == "regime", "segment"]) == {"STABLE", "VOLATILE"}

    target_report = pd.read_csv(target_metrics_path)
    assert set(target_report.loc[target_report["scope"] == "competition", "segment"]) == {"EPL", "UCL"}
    assert set(target_report.loc[target_report["scope"] == "phase", "segment"]) == {"REGULAR", "KNOCKOUT"}

