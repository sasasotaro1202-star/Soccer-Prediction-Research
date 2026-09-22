import json

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
    _write(tmp_path / "score_oos_gate.json", {
        "status": "PASS",
        "blocks": 4,
        "development_blocks": 2,
        "locked_blocks": 2,
        "minimum_development_blocks": 2,
        "minimum_locked_blocks": 2,
        "chronological_split_valid": True,
        "rows": 10,
        "finite_metrics": True,
    })
    _write(tmp_path / "score_model_selection.json", {
        "selected_method": "primary",
        "selection_rule": {"locked_oos_inspected": False},
    })
    _write(tmp_path / "score_locked_gate.json", {
        "selected_method": "primary",
        "status": "PASS",
    })
    _write(tmp_path / "candidate_lock.json", {"locked_oos_untouched": True, "locked_oos_blocks": 2})
    _write(tmp_path / "adoption_decision.json", {
        "status": "ADOPT",
        "oos_claimed": True,
        "stability": {"status": "PASS"},
        "external_stability_gate": {"status": "PASS"},
    })
    # The contract only needs a non-empty bundle in the unit fixture; runtime
    # integration tests validate that the real pickle is loadable elsewhere.
    (tmp_path / "production_model.pkl").write_bytes(b"test-bundle")
    _write(tmp_path / "model_registry.json", {"adoption_status": "ADOPT"})
    _write(tmp_path / "production_model.json", {"adoption_status": "ADOPT"})


def test_contract_fails_closed_when_evidence_is_missing(tmp_path):
    result = evaluate_production_contract(str(tmp_path))
    assert result.passed is False
    assert "completion_gate" in result.failures
    assert "oos_claim" in result.failures
    assert "tests" in result.failures
    assert "audit_gate" in result.failures


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
