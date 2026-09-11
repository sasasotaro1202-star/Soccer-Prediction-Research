from __future__ import annotations

import json
from pathlib import Path


def load_adopted_model(registry_path: str = "artifacts/model_registry.json"):
    p = Path(registry_path)
    if not p.exists():
        raise RuntimeError("No adopted model registry exists; production prediction is fail-closed")
    record = json.loads(p.read_text(encoding="utf-8"))
    if record.get("adoption_status") != "ADOPT":
        raise RuntimeError("Registry contains no ADOPT model")
    return record


def run():
    return load_adopted_model()
