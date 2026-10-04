import json

from src.research.registry import load_champion, save_registry


def test_adopted_registry_record_is_persisted_as_active(tmp_path):
    path = tmp_path / "model_registry.json"
    record = save_registry(
        str(path),
        model_version="v1",
        feature_version="pit_safe_v1",
        research_cycle="cycle-1",
        git_commit_sha="sha-1",
        data_snapshot_id="snapshot-1",
        metrics={},
        adoption_status="ADOPT",
        parameters={"weights": {"logistic": 1.0}},
        calibration={"temperature": 1.0},
        feature_cols=["elo_diff", "form_5"],
        feature_manifest_version="soccer-feature-contract-v1",
        feature_policy_version="soccer-feature-policy-v1",
        feature_schema_hash="schema-hash",
        target_version="soccer-target-contract-v1",
    )
    assert record["adoption_status"] == "ADOPT"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["adoption_status"] == "ADOPT"
    assert payload["feature_cols"] == ["elo_diff", "form_5"]
    assert payload["feature_manifest_version"] == "soccer-feature-contract-v1"
    assert payload["feature_policy_version"] == "soccer-feature-policy-v1"
    assert payload["feature_schema_hash"] == "schema-hash"
    assert payload["target_version"] == "soccer-target-contract-v1"
    assert load_champion(str(path))["model_version"] == "v1"
