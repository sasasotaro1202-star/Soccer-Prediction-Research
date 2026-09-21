from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_registry(
    path: str,
    model_version: str,
    feature_version: str,
    research_cycle: str,
    git_commit_sha: str,
    data_snapshot_id: str,
    metrics: dict,
    adoption_status: str,
    parameters: dict[str, Any] | None = None,
    training_end: str | None = None,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a reproducible registry record and preserve prior history."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "model_version": model_version,
        "feature_version": feature_version,
        "research_cycle": research_cycle,
        "git_commit_sha": git_commit_sha,
        "data_snapshot_id": data_snapshot_id,
        "metrics": metrics,
        "parameters": parameters or {},
        "training_end": training_end,
        "calibration": calibration or {},
        "adoption_status": adoption_status,
        "oos_verified": adoption_status.upper() in {"ADOPT", "CHAMPION", "ADOPTED"},
        "created_at_utc": _utc_now(),
    }
    history_path = p.with_name(p.stem + "_history.jsonl")
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    if adoption_status.upper() in {"ADOPT", "CHAMPION", "ADOPTED"}:
        p.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    elif not p.exists():
        p.write_text(json.dumps({"status": "NO_CHAMPION", "latest_candidate": record}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return record


def load_champion(path: str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("adoption_status", "").upper() not in {"ADOPT", "CHAMPION", "ADOPTED"}:
        return None
    return data


def rollback_to(path: str, history_path: str, model_version: str) -> dict[str, Any]:
    """Restore a prior champion from immutable JSONL history."""
    target = None
    hp = Path(history_path)
    if hp.exists():
        for line in hp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("model_version") == model_version:
                target = item
    if target is None:
        raise ValueError(f"model_version not found in registry history: {model_version}")
    target = dict(target)
    target["adoption_status"] = "CHAMPION"
    target["rollback_at_utc"] = _utc_now()
    Path(path).write_text(json.dumps(target, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return target
