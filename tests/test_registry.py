from __future__ import annotations

import json

from src.research.registry import load_champion, save_registry


def test_adopt_status_is_persisted_as_active_registry(tmp_path):
    path = tmp_path / "model_registry.json"
    record = save_registry(
        str(path),
        model_version="v1",
        feature_version="pit_safe_v1",
        research_cycle="cycle-1",
        git_commit_sha="sha",
        data_snapshot_id="snapshot",
        metrics={"locked_oos": []},
        adoption_status="ADOPT",
    )
    assert record["adoption_status"] == "ADOPT"
    stored = json.loads(path.read_text())
    assert stored["adoption_status"] == "ADOPT"
    assert stored["oos_verified"] is True
    assert load_champion(str(path))["model_version"] == "v1"


def test_champion_status_remains_loadable(tmp_path):
    path = tmp_path / "model_registry.json"
    save_registry(
        str(path),
        model_version="v2",
        feature_version="pit_safe_v1",
        research_cycle="cycle-2",
        git_commit_sha="sha",
        data_snapshot_id="snapshot",
        metrics={"locked_oos": []},
        adoption_status="CHAMPION",
    )
    assert load_champion(str(path))["model_version"] == "v2"
