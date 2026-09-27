"""Immutable research checkpoint ledger with success-only reuse.

Inspired by the cross-project research caches: failures remain as evidence,
but only an exact successful checkpoint (same experiment key, commit SHA and
configuration hash) can be reused. Existing records are never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SUCCESS = "SUCCESS"
FAILURE = "FAILURE"
BLOCKED = "BLOCKED"
CANCELLED = "CANCELLED"
_ALLOWED = {SUCCESS, FAILURE, BLOCKED, CANCELLED}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: Any) -> str:
    return hashlib.sha256(_canonical(config).encode("utf-8")).hexdigest()


def make_checkpoint(
    *,
    experiment_key: str,
    status: str,
    git_sha: str,
    run_id: str | int,
    configuration: Any,
    artifacts: Iterable[str] = (),
    metrics: dict[str, Any] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    status = str(status).upper()
    if status not in _ALLOWED:
        raise ValueError(f"invalid checkpoint status: {status}")
    if not experiment_key or not git_sha:
        raise ValueError("experiment_key and git_sha are required")
    return {
        "schema_version": 1,
        "experiment_key": str(experiment_key),
        "status": status,
        "git_sha": str(git_sha),
        "run_id": str(run_id),
        "config_hash": config_hash(configuration),
        "configuration": configuration,
        "artifacts": sorted({str(x) for x in artifacts if x}),
        "metrics": dict(metrics or {}),
        "reason": reason,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def read_checkpoints(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid checkpoint JSON at line {line_no}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"checkpoint line {line_no} is not an object")
        rows.append(value)
    return rows


def append_checkpoint(path: str | Path, checkpoint: dict[str, Any]) -> str:
    """Append without mutation; return INSERTED or EXISTING_SUCCESS."""
    required = {"schema_version", "experiment_key", "status", "git_sha", "config_hash"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"checkpoint missing keys: {missing}")
    status = str(checkpoint["status"]).upper()
    if status not in _ALLOWED:
        raise ValueError(f"invalid checkpoint status: {status}")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    existing = read_checkpoints(p)
    exact = [
        row for row in existing
        if row.get("experiment_key") == checkpoint["experiment_key"]
        and row.get("git_sha") == checkpoint["git_sha"]
        and row.get("config_hash") == checkpoint["config_hash"]
    ]
    if any(str(row.get("status", "")).upper() == SUCCESS for row in exact):
        return "EXISTING_SUCCESS"

    with p.open("a", encoding="utf-8") as fh:
        fh.write(_canonical(checkpoint) + "\n")
    return "INSERTED"


def latest_success(
    path: str | Path,
    *,
    experiment_key: str,
    git_sha: str,
    configuration: Any,
) -> dict[str, Any] | None:
    wanted = config_hash(configuration)
    rows = read_checkpoints(path)
    matches = [
        row for row in rows
        if row.get("experiment_key") == experiment_key
        and row.get("git_sha") == git_sha
        and row.get("config_hash") == wanted
        and str(row.get("status", "")).upper() == SUCCESS
    ]
    return matches[-1] if matches else None


def can_reuse_success(path: str | Path, *, experiment_key: str, git_sha: str, configuration: Any) -> bool:
    return latest_success(
        path,
        experiment_key=experiment_key,
        git_sha=git_sha,
        configuration=configuration,
    ) is not None


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--store", required=True)
    rec.add_argument("--experiment-key", required=True)
    rec.add_argument("--status", required=True)
    rec.add_argument("--git-sha", required=True)
    rec.add_argument("--run-id", required=True)
    rec.add_argument("--config-json", required=True)
    rec.add_argument("--artifacts-json", default="[]")
    rec.add_argument("--metrics-json", default="{}")
    rec.add_argument("--reason", default=None)
    args = parser.parse_args()
    if args.command == "record":
        checkpoint = make_checkpoint(
            experiment_key=args.experiment_key,
            status=args.status,
            git_sha=args.git_sha,
            run_id=args.run_id,
            configuration=json.loads(args.config_json),
            artifacts=json.loads(args.artifacts_json),
            metrics=json.loads(args.metrics_json),
            reason=args.reason,
        )
        result = append_checkpoint(args.store, checkpoint)
        print(json.dumps({"status": result, "checkpoint": checkpoint}, ensure_ascii=False))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
