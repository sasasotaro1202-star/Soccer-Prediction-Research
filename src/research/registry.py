from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def save_registry(path: str, model_version: str, feature_version: str, research_cycle: str, git_commit_sha: str, data_snapshot_id: str, metrics: dict, adoption_status: str):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "model_version": model_version,
        "feature_version": feature_version,
        "research_cycle": research_cycle,
        "git_commit_sha": git_commit_sha,
        "data_snapshot_id": data_snapshot_id,
        "metrics": metrics,
        "adoption_status": adoption_status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    p.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return record
