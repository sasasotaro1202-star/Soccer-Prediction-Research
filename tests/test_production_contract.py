import json

from src.research.production_contract import evaluate_production_contract, write_contract_result


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _minimal_passing_artifacts(tmp_path):
    # Production contract expects artifacts to live under the supplied root.
    # Use a real two-line CSV so the test exercises the same non-empty check as production.
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
    ):
        (tmp_path / name).write_text("metric,value\naccuracy,0.8\n", encoding="utf-8")
    _write(tmp_path / "candidate_lock.json", {"locked_oos_untouched": True, "locked_oos_blocks": 2})
    _write(tmp_path / "adoption_decision.json", {
        "status": "ADOPT",
        "oos_claimed": True,
        "stability": {"status": "PASS"},
    })


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
