import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import experience_ledger as mod


def _base_row(**extra):
    row = {
        "match_id": "espn:test",
        "kickoff_utc": "2026-09-26T10:00:00Z",
        "home_team": "A",
        "away_team": "B",
        "competition": "EPL",
        "p_home": 0.5,
        "p_draw": 0.25,
        "p_away": 0.25,
        "model_version": "v1",
    }
    row.update(extra)
    return row


def test_record_requires_explicit_prediction_time_when_column_is_absent(tmp_path: Path, monkeypatch):
    ledger = tmp_path / "ledger.csv"
    snapshots = tmp_path / "snapshots.jsonl"
    predictions = tmp_path / "predictions.csv"
    monkeypatch.setattr(mod, "LEDGER", ledger)
    monkeypatch.setattr(mod, "PREDICTION_SNAPSHOTS", snapshots)
    pd.DataFrame([_base_row()]).to_csv(predictions, index=False)
    with pytest.raises(RuntimeError, match="explicit prediction_time_utc"):
        mod.record_prediction_file(str(predictions))


def test_record_rejects_prediction_at_or_after_kickoff(tmp_path: Path, monkeypatch):
    ledger = tmp_path / "ledger.csv"
    snapshots = tmp_path / "snapshots.jsonl"
    predictions = tmp_path / "predictions.csv"
    monkeypatch.setattr(mod, "LEDGER", ledger)
    monkeypatch.setattr(mod, "PREDICTION_SNAPSHOTS", snapshots)
    pd.DataFrame([_base_row(
        prediction_time_utc="2026-09-26T11:00:00Z",
    )]).to_csv(predictions, index=False)
    with pytest.raises(RuntimeError, match="at/after kickoff"):
        mod.record_prediction_file(str(predictions))


def test_settlement_requires_pit_validated_prediction():
    row = pd.Series(_base_row(
        fixture_key="unused",
        prediction_pit_cutoff_utc="2026-09-26T08:00:00+00:00",
    ))
    assert mod._settle_row(row, {}, {}) == {}
