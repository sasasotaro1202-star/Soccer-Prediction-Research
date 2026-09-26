from __future__ import annotations

import json

import pandas as pd

from src.research.v13_ultimate_controls import complete_v13


def _write_required(root):
    (root / "experiment_results.json").write_text(
        json.dumps({
            "status": "COMPLETED_RESEARCH_ONLY",
            "oos_claimed": True,
            "production_changed": False,
            "pit_audit": {"status": "PASS"},
            "meta_leakage_audit": {"status": "PASS"},
            "leakage_audit": {"audit_status": "PARTIAL"},
            "promotion": {"status": "HOLD"},
        }),
        encoding="utf-8",
    )
    (root / "experiment_manifest.json").write_text(
        json.dumps({"experiment_id": "test-v13", "git_sha": "abc"}),
        encoding="utf-8",
    )
    (root / "promotion_gate.json").write_text(json.dumps({"status": "HOLD"}), encoding="utf-8")
    (root / "meta_leakage_audit.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    (root / "leakage_audit.json").write_text(json.dumps({"audit_status": "PARTIAL"}), encoding="utf-8")

    pd.DataFrame([{
        "block": 0,
        "accuracy": 0.50,
        "logloss": 1.00,
        "brier": 0.68,
        "ece": 0.10,
        "baseline_accuracy": 0.50,
        "baseline_logloss": 1.01,
        "baseline_brier": 0.69,
        "baseline_ece": 0.11,
        "delta_accuracy": 0.00,
        "delta_logloss": -0.01,
        "delta_brier": -0.01,
        "delta_ece": -0.01,
    }]).to_csv(root / "block_metrics.csv", index=False)

    pd.DataFrame([{"variant": "baseline", "accuracy": 0.50}]).to_csv(root / "ablation_results.csv", index=False)
    pd.DataFrame([{"model_a": "a", "model_b": "b", "error_correlation": 0.0}]).to_csv(root / "error_correlation.csv", index=False)
    pd.DataFrame([{"total_uncertainty": 0.2, "update_need": 0.1}]).to_csv(root / "uncertainty_by_row.csv", index=False)
    pd.DataFrame([{
        "block": 0,
        "match_id": "m1",
        "kickoff_utc": "2026-01-01T12:00:00Z",
        "prediction": 0,
        "future_prediction": 1,
        "target": 1,
        "confidence": 0.62,
        "predictability": 0.70,
        "uncertainty": 0.20,
        "flip_risk": 0.10,
        "action": "PREDICT",
        "strategy": "ensemble",
    }]).to_csv(root / "prediction_policy_and_output.csv", index=False)
    (root / "future_failure_by_block.json").write_text("[]", encoding="utf-8")
    pd.DataFrame([{"block": 0, "stress": "noise", "logloss": 1.0}]).to_csv(root / "robustness_stress.csv", index=False)


def test_complete_v13_builds_auditable_outputs(tmp_path):
    _write_required(tmp_path)

    result = complete_v13(str(tmp_path))

    assert result["status"] == "COMPLETED_RESEARCH_ONLY"
    assert result["leakage"] == "FAIL"
    assert result["promotion"] == "HOLD"
    assert (tmp_path / "prediction_history.csv").exists()
    assert (tmp_path / "revision_analysis.csv").exists()
    assert (tmp_path / "strategy_failure.csv").exists()
    assert (tmp_path / "health_monitor.json").exists()
    assert (tmp_path / "fallback_plan.json").exists()
    assert (tmp_path / "nested_oos_status.json").exists()
    assert (tmp_path / "router_stability.json").exists()
    assert (tmp_path / "forecast_contract.csv").exists()
    assert (tmp_path / "reproducibility.json").exists()
    assert (tmp_path / "kill_switch.json").exists()
    assert (tmp_path / "state_matrix.json").exists()

    revision = pd.read_csv(tmp_path / "revision_analysis.csv")
    assert bool(revision.loc[0, "revision_event"]) is True
    assert bool(revision.loc[0, "revision_improvement"]) is True
    assert bool(revision.loc[0, "false_revision"]) is False


def test_complete_v13_fails_closed_when_artifact_missing(tmp_path):
    (tmp_path / "experiment_results.json").write_text("{}", encoding="utf-8")
    result = complete_v13(str(tmp_path))
    assert result["status"] == "BLOCKED"
    assert "block_metrics.csv" in result["missing"]
