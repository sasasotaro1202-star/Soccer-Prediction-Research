import json
import os

from src.research.production_contract import evaluate_production_contract, write_contract_result


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _minimal_passing_artifacts(tmp_path):
    _write(tmp_path / "completion_gate.json", {"full_gate_passed": True, "pit_publication_time_gate": True})
    _write(tmp_path / "test_status.json", {"passed": True, "exit_code": 0})
    _write(tmp_path / "audit_status.json", {"passed": True, "exit_code": 0})
    _write(tmp_path / "audit_gate.json", {"full_gate_passed": True})
    _write(tmp_path / "run_status.json", {
        "status": "READY",
        "oos_claimed": True,
        "gates": {name: True for name in (
            "data", "schema", "leakage", "features", "training",
            "backtest", "oos", "prediction", "sanity", "artifact"
        )},
    })
    for name in (
        "oos_metrics.csv", "model_selection.csv",
        "development_oos_metrics.csv", "locked_oos_metrics.csv",
        "score_oos_metrics.csv", "score_oos_gate.json",
        "score_model_selection.json", "score_locked_gate.json",
    ):
        (tmp_path / name).write_text("metric,value\nplaceholder,1\n", encoding="utf-8")
    _write(tmp_path / "score_oos_gate.json", {"status": "PASS", "blocks": 5, "minimum_total_blocks": 5, "minimum_rows_per_block": 500, "block_rows": [500, 500, 500, 500, 500], "block_rows_ok": True, "rows": 2500, "finite_metrics": True})
    _write(tmp_path / "score_model_selection.json", {
        "selected_method": "primary",
        "selection_rule": {"locked_oos_inspected": False},
    })
    _write(tmp_path / "oos_temporal_integrity.json", {"status": "PASS", "fail_closed": True})
    _write(tmp_path / "score_oos_temporal_integrity.json", {"status": "PASS", "fail_closed": True})
    _write(tmp_path / "score_locked_gate.json", {
        "selected_method": "primary",
        "status": "PASS",
        "locked_oos_inspected": True,
        "baseline_metrics": {
            "score_logloss": 1.0,
            "over_2_5_logloss": 0.7,
            "over_2_5_brier": 0.2,
            "btts_logloss": 0.68,
            "btts_brier": 0.21,
        },
        "locked_block_rows": [500, 500],
        "minimum_rows_per_block": 500,
        "selected_metrics": {
            "score_logloss": 1.0,
            "over_2_5_logloss": 0.7,
            "over_2_5_brier": 0.2,
            "btts_logloss": 0.68,
            "btts_brier": 0.21,
        },
    })
    _write(tmp_path / "candidate_lock.json", {"locked_oos_untouched": True, "locked_oos_blocks": 2})
    _write(tmp_path / "adoption_decision.json", {
        "status": "ADOPT",
        "oos_claimed": True,
        "stability": {"status": "PASS"},
        "external_stability_gate": {"status": "PASS"},
        "locked_block_rows": [500, 500],
        "minimum_locked_rows_per_block": 500,
    })
    # The contract only needs a non-empty bundle in the unit fixture; runtime
    # integration tests validate that the real pickle is loadable elsewhere.
    (tmp_path / "production_model.pkl").write_bytes(b"test-bundle")
    _write(tmp_path / "model_registry.json", {
        "adoption_status": "ADOPT",
        "git_commit_sha": os.getenv("GITHUB_SHA", ""),
    })
    _write(tmp_path / "production_model.json", {"adoption_status": "ADOPT"})


def test_contract_fails_closed_when_evidence_is_missing(tmp_path):
    result = evaluate_production_contract(str(tmp_path))
    assert result.passed is False
    assert "completion_gate" in result.failures
    assert "oos_claim" in result.failures
    assert "tests" in result.failures
    assert "audit_gate" in result.failures


def test_contract_rejects_insufficient_score_oos_depth(tmp_path):
    _minimal_passing_artifacts(tmp_path)
    _write(tmp_path / "score_oos_gate.json", {"status": "PASS", "blocks": 4, "minimum_total_blocks": 5, "rows": 100, "finite_metrics": True})
    result = evaluate_production_contract(str(tmp_path))
    assert result.passed is False
    assert "score_oos_blocks" in result.failures


def test_contract_passes_only_with_explicit_success(tmp_path):
    _minimal_passing_artifacts(tmp_path)
    result = write_contract_result(str(tmp_path))
    assert result.passed is True
    payload = json.loads((tmp_path / "production_contract.json").read_text())
    assert payload["production_contract_passed"] is True
    assert payload["fail_closed"] is True


def test_hold_adoption_cannot_pass(tmp_path):
    _minimal_passing_artifacts(tmp_path)
    _write(tmp_path / "adoption_decision.json", {"status": "HOLD", "oos_claimed": True, "stability": {"status": "PASS"}})
    result = evaluate_production_contract(str(tmp_path))
    assert result.passed is False
    assert "adoption:HOLD" in result.failures


def test_missing_stability_evidence_cannot_pass(tmp_path):
    _minimal_passing_artifacts(tmp_path)
    _write(tmp_path / "adoption_decision.json", {"status": "ADOPT", "oos_claimed": True})
    result = evaluate_production_contract(str(tmp_path))
    assert result.passed is False
    assert "adoption_stability" in result.failures


def test_blocked_status_cannot_pass(tmp_path):
    _minimal_passing_artifacts(tmp_path)
    _write(tmp_path / "run_status.json", {"status": "BLOCKED", "oos_claimed": True})
    result = evaluate_production_contract(str(tmp_path))
    assert result.passed is False
    assert any(x.startswith("run_status:") for x in result.failures)


def test_stale_git_commit_provenance_cannot_pass(tmp_path, monkeypatch):
    _minimal_passing_artifacts(tmp_path)
    _write(tmp_path / "model_registry.json", {
        "adoption_status": "ADOPT",
        "model_version": "v1",
        "git_commit_sha": "old-commit",
    })
    monkeypatch.setenv("GITHUB_SHA", "new-commit")
    result = evaluate_production_contract(str(tmp_path))
    assert result.passed is False
    assert "git_commit_provenance_mismatch" in result.failures


def test_contract_rejects_tampered_production_artifact_after_provenance(tmp_path):
    _minimal_passing_artifacts(tmp_path)
    write_contract_result(str(tmp_path))
    (tmp_path / "production_model.json").write_text(
        json.dumps({"adoption_status": "ADOPT", "tampered": True}),
        encoding="utf-8",
    )
    result = evaluate_production_contract(str(tmp_path))
    assert result.passed is False
    assert "provenance_hash_mismatch:production_model.json" in result.failures


def test_production_contract_rejects_legacy_adopted_bundle_without_routing(monkeypatch, tmp_path):
    from src.research.production_contract import write_contract_result

    payloads = {
        "completion_gate.json": {"full_gate_passed": True, "pit_publication_time_gate": True},
        "test_status.json": {"passed": True},
        "audit_status.json": {"passed": True},
        "audit_gate.json": {"full_gate_passed": True},
        "run_status.json": {"oos_claimed": True, "status": "OK", "gates": {name: True for name in (
            "data", "schema", "leakage", "features", "training", "backtest", "oos",
            "prediction", "sanity", "artifact",
        )}},
        "adoption_decision.json": {
            "status": "ADOPT",
            "oos_claimed": True,
            "stability": {"status": "PASS"},
            "external_stability_gate": {"status": "PASS"},
            "locked_block_rows": [500, 500],
            "minimum_locked_rows_per_block": 500,
        },
        "score_oos_gate.json": {
            "status": "PASS", "blocks": 5, "minimum_total_blocks": 5, "rows": 2500,
            "minimum_rows_per_block": 500, "block_rows": [500, 500, 500, 500, 500],
            "block_rows_ok": True, "finite_metrics": True,
        },
        "score_model_selection.json": {"selected_method": "primary", "selection_rule": {"locked_oos_inspected": False}},
        "score_locked_gate.json": {
            "status": "PASS", "selected_method": "primary", "locked_oos_inspected": True,
            "locked_block_rows": [500, 500], "minimum_rows_per_block": 500,
            "selected_metrics": {
                "score_logloss": 1.0, "over_2_5_logloss": 0.7, "over_2_5_brier": 0.2,
                "btts_logloss": 0.7, "btts_brier": 0.2,
            },
        },
        "oos_temporal_integrity.json": {"status": "PASS"},
        "score_oos_temporal_integrity.json": {"status": "PASS"},
        "candidate_lock.json": {"locked_oos_untouched": True, "locked_oos_blocks": 2},
        "production_model.json": {"adoption_status": "ADOPT", "model_version": "v", "feature_cols": ["f1"]},
        "model_registry.json": {"adoption_status": "ADOPT", "model_version": "v", "feature_cols": ["f1"], "git_commit_sha": "unknown"},
    }
    for name, value in payloads.items():
        (tmp_path / name).write_text(__import__("json").dumps(value), encoding="utf-8")
    (tmp_path / "oos_metrics.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "model_selection.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "development_oos_metrics.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "locked_oos_metrics.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "score_oos_metrics.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "production_model.pkl").write_bytes(b"not-a-real-model")

    result = write_contract_result(str(tmp_path))
    assert result.passed is False
    assert "production_bundle_load" in result.failures
