import json

import numpy as np
import pandas as pd

from src.research.engine import _build_research_gates


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_artifacts(tmp_path):
    _write_json(
        tmp_path / "completion_gate.json",
        {"full_gate_passed": True, "pit_publication_time_gate": True},
    )
    _write_json(tmp_path / "audit_gate.json", {"full_gate_passed": True})
    (tmp_path / "pit_replay_features.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "score_oos_gate.json").write_text(
        json.dumps({"status": "PASS", "blocks": 3, "rows": 6000, "finite_metrics": True}),
        encoding="utf-8",
    )
    (tmp_path / "score_locked_gate.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )
    for name in (
        "oos_metrics.csv",
        "model_selection.csv",
        "development_oos_metrics.csv",
        "locked_oos_metrics.csv",
        "score_oos_metrics.csv",
        "score_model_selection.json",
        "candidate_lock.json",
        "adoption_decision.json",
    ):
        (tmp_path / name).write_text("x,y\n1,2\n", encoding="utf-8")


def _frames():
    wf = pd.DataFrame(
        {
            "logloss": [0.9, 0.92, 0.91],
            "brier": [0.18, 0.19, 0.185],
            "n": [1000, 1000, 1000],
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
