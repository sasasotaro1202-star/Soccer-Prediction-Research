from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone


def build_manifest(data_snapshot_id: str, configuration: dict, oos_split: dict, prediction_cutoff: str | None = None) -> dict:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        sha = "unknown"
    payload = {"git_commit_sha": sha, "data_snapshot_id": data_snapshot_id, "configuration": configuration, "oos_split": oos_split, "prediction_cutoff": prediction_cutoff}
    payload["manifest_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    return payload


def save_manifest(path: str, manifest: dict):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
