import json

import numpy as np
import pandas as pd

from src.research.engine import _build_research_gates, _build_stability_folds


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_artifacts(tmp_path):
    _write_json(
        tmp_path / "completion_gate.json",
        {"full_gate_passed": True, "pit_publication_time_gate": True},
    )
    _write_json(tmp_path / "audit_gate.json", {"full_gate_passed": True})
    (tmp_path / "pit_replay_features.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "production_model.pkl").write_bytes(b"bundle")
    (tmp_path / "score_oos_gate.json").write_text(
        json.dumps({"status": "PASS", "blocks": 5, "minimum_total_blocks": 5, "minimum_rows_per_block": 500, "block_rows": [1000, 1000, 1000, 1000, 1000], "block_rows_ok": True, "rows": 6000, "finite_metrics": True}),
        encoding="utf-8",
    )
    (tmp_path / "score_locked_gate.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )
    _write_json(
        tmp_path / "oos_temporal_integrity.json",
        {"status": "PASS", "fail_closed": True},
    )
    _write_json(
        tmp_path / "score_oos_temporal_integrity.json",
        {"status": "PASS", "fail_closed": True},
    )
    _write_json(tmp_path / "calibration_gate.json", {"status": "PASS", "temperature": 1.0, "calibration_rows": 120, "locked_oos_used_for_calibration": False})
    for name in (
        "oos_metrics.csv",
        "model_selection.csv",
        "development_oos_metrics.csv",
        "locked_oos_metrics.csv",
        "score_oos_metrics.csv",
        "score_model_selection.json",
        "score_model_selection_by_competition.json",
        "score_locked_gate_by_competition.json",
        "candidate_lock.json",
        "adoption_decision.json",
    ):
        if name.endswith(".json"):
            _write_json(tmp_path / name, {"status": "PASS"})
        else:
            (tmp_path / name).write_text("x,y\n1,2\n", encoding="utf-8")


def _frames():
    wf = pd.DataFrame(
        {
            "logloss": [0.9, 0.92, 0.91],
            "brier": [0.18, 0.19, 0.185],
            "n": [1000, 1000, 1000],
            "oos_start": ["2020-01-01", "2020-06-01", "2021-01-01"],
            "oos_end": ["2020-05-31", "2020-12-31", "2021-06-30"],
        }
    )
    development = wf.iloc[:1].copy()
    locked = wf.iloc[1:].copy()
    selections = pd.DataFrame([{"selected_model": "logistic", "weights": {"logistic": 1.0}}])
    stability = {"status": "PASS"}
    return wf, development, locked, selections, stability


def test_build_research_gates_emits_complete_contract_map(tmp_path):
    _make_artifacts(tmp_path)
    wf, development, locked, selections, stability = _frames()
    gates = _build_research_gates(
        tmp_path,
        history_nonempty=True,
        pit_verified=6000,
        wf=wf,
        development_oos=development,
        locked_oos=locked,
        selections=selections,
        stability=stability,
        adoption={"status": "ADOPT"},
        model_bundle={"status": "READY"},
        audit_report={"audit_execution_ok": True},
    )
    assert set(gates) == {
        "data", "schema", "leakage", "features", "training",
        "backtest", "oos", "prediction", "sanity", "artifact",
    }
    assert all(gates.values())


def test_build_research_gates_blocks_prediction_without_adoption(tmp_path):
    _make_artifacts(tmp_path)
    wf, development, locked, selections, stability = _frames()
    gates = _build_research_gates(
        tmp_path,
        history_nonempty=True,
        pit_verified=6000,
        wf=wf,
        development_oos=development,
        locked_oos=locked,
        selections=selections,
        stability=stability,
        adoption={"status": "HOLD"},
        model_bundle=None,
        audit_report={"audit_execution_ok": True},
    )
    assert gates["oos"] is True
    assert gates["prediction"] is False


def test_stability_fold_builder_accepts_development_oos_only():
    frame = pd.DataFrame(
        {
            "leagues": ["EPL", "Bundesliga", "Serie A"],
            "seasons": ["2022", "2023", "2024"],
            "oos_start": ["2024-01-01", "2024-02-01", "2024-03-01"],
            "oos_end": ["2024-01-31", "2024-02-29", "2024-03-31"],
            "baseline_logistic_logloss": [1.0, 1.0, 1.0],
            "baseline_logistic_brier": [0.25, 0.25, 0.25],
            "baseline_logistic_accuracy": [0.50, 0.50, 0.50],
            "logloss": [0.95, 0.96, 0.97],
            "brier": [0.24, 0.24, 0.24],
            "accuracy": [0.51, 0.51, 0.51],
        }
    )
    development = frame.iloc[:2].copy()
    locked = frame.iloc[2:].copy()

    folds = _build_stability_folds(development)

    assert len(folds) == len(development)
    assert len(folds) != len(frame)
    assert folds[-1]["oos_end_utc"] == "2024-02-29"
    assert all(f["oos_start_utc"] != locked.iloc[0]["oos_start"] for f in folds)
    assert all(f["oos_end_utc"] != locked.iloc[0]["oos_end"] for f in folds)
