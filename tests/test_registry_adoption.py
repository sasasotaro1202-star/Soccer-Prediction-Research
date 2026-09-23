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
    )
    assert record["adoption_status"] == "ADOPT"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["adoption_status"] == "ADOPT"
    assert load_champion(str(path))["model_version"] == "v1"
