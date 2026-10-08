from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


TASK_FILES = {
    "1X2": ("locked_oos_metrics.csv", "oos_metrics.csv"),
    "Score": ("score_locked_oos_metrics.csv", "score_oos_metrics.csv"),
    "O/U": ("ou_locked_oos_metrics.csv", "ou_oos_metrics.csv"),
    "BTTS": ("btts_locked_oos_metrics.csv", "btts_oos_metrics.csv"),
    "MOM": ("mom_locked_oos_metrics.csv", "mom_oos_metrics.csv"),
}

TASK_METRICS = {
    "1X2": ("logloss", "accuracy", "brier", "rps", "ece"),
    "Score": (
        "score_logloss",
        "exact_score_hit_rate",
        "top3_score_hit_rate",
        "top4_score_hit_rate",
        "home_goals_mae",
        "away_goals_mae",
        "total_goals_mae",
    ),
    "O/U": ("logloss", "accuracy", "brier", "ece"),
    "BTTS": ("logloss", "accuracy", "brier", "ece"),
    "MOM": ("top1_accuracy", "top4_hit_rate", "mrr", "logloss", "brier", "ece"),
}


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def _num(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


def _weighted_mean(rows: list[dict[str, str]], field: str) -> float | None:
    values: list[tuple[float, float]] = []
    for row in rows:
        value = _num(row.get(field))
        weight = _num(row.get("n")) or 0.0
        if value is None:
            continue
        values.append((value, max(weight, 1.0)))
    if not values:
        return None
    total = sum(weight for _, weight in values)
    return sum(value * weight for value, weight in values) / total


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gate_state(root: Path) -> tuple[str, list[str]]:
    completion = _json(root / "completion_gate.json")
    audit = _json(root / "audit_gate.json")
    reasons: list[str] = []

    if completion:
        if completion.get("full_gate_passed") is not True:
            reasons.extend(str(x) for x in completion.get("blocking_reasons", []) if str(x))
    else:
        reasons.append("completion_gate_artifact_missing")

    if audit:
        if audit.get("full_gate_passed") is not True:
            reasons.extend(str(x) for x in audit.get("blocking_reasons", []) if str(x))
    else:
        reasons.append("audit_gate_artifact_missing")

    if reasons:
        return "BLOCKED", sorted(set(reasons))
    return "PASS", []


def _adoption_status(root: Path) -> str:
    adoption = _json(root / "adoption_decision.json")
    status = str(adoption.get("status", "")).strip().upper()
    return status or "UNKNOWN"


def _task_record(root: Path, task: str, gate_state: str, gate_reasons: list[str]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "task": task,
        "status": "BLOCKED" if gate_state == "BLOCKED" else "UNVERIFIED",
        "source_file": None,
        "blocks": 0,
        "n": 0,
        "metrics": {},
    }
    if gate_state == "BLOCKED":
        record["blocking_reasons"] = gate_reasons
        return record

    integrity_name = "oos_temporal_integrity.json" if task == "1X2" else "score_oos_temporal_integrity.json"
    integrity = _json(root / integrity_name)
    if integrity.get("status") != "PASS":
        record["reason"] = f"Chronological OOS integrity not verified: {integrity_name}"
        record["integrity_status"] = str(integrity.get("status") or "UNKNOWN")
        return record

    # O/U and BTTS must be independently evidenced. Score-model-derived fields
    # are intentionally not relabeled as standalone target performance.
    for filename in TASK_FILES[task]:
        path = root / filename
        rows = _csv(path)
        if not rows:
            continue
        metrics = {}
        for field in TASK_METRICS[task]:
            value = _weighted_mean(rows, field)
            if value is not None:
                metrics[field] = value
        if metrics:
            record["status"] = "EVALUATED"
            record["source_file"] = filename
            record["blocks"] = len(rows)
            record["n"] = int(sum((_num(row.get("n")) or 0.0) for row in rows))
            record["metrics"] = {key: float(value) for key, value in metrics.items()}
            break

    if record["status"] == "UNVERIFIED":
        record["reason"] = "No target-specific OOS metric artifact was found."
    return record


def build_snapshot(root: Path) -> dict[str, Any]:
    root = Path(root)
    gate_state, gate_reasons = _gate_state(root)
    integrity_oos = _json(root / "oos_temporal_integrity.json")
    integrity_score = _json(root / "score_oos_temporal_integrity.json")
    calibration = _json(root / "calibration_gate.json")
    adoption = _adoption_status(root)

    tasks = [
        _task_record(root, task, gate_state, gate_reasons)
        for task in ("1X2", "Score", "O/U", "BTTS", "MOM")
    ]

    source_paths: list[str] = []
    for task in tasks:
        if task.get("source_file"):
            source_paths.append(str(root / str(task["source_file"])))
    for name in (
        "completion_gate.json",
        "audit_gate.json",
        "oos_temporal_integrity.json",
        "score_oos_temporal_integrity.json",
        "calibration_gate.json",
        "adoption_decision.json",
        "target_oos_status.json",
    ):
        path = root / name
        if path.is_file():
            source_paths.append(str(path))

    evidence_hashes = {}
    for raw in sorted(set(source_paths)):
        path = Path(raw)
        if path.is_file():
            evidence_hashes[path.name] = _sha256(path)

    overall = "BLOCKED" if gate_state == "BLOCKED" else (
        "ADOPTED" if adoption == "ADOPT" else
        "EVALUATED" if any(item["status"] == "EVALUATED" for item in tasks) else
        "NO_VERIFIED_PERFORMANCE"
    )

    return {
        "schema_version": 1,
        "status": overall,
        "pit_gate": gate_state,
        "pit_blocking_reasons": gate_reasons,
        "adoption_status": adoption,
        "oos_temporal_integrity": integrity_oos or {"status": "UNKNOWN"},
        "score_oos_temporal_integrity": integrity_score or {"status": "UNKNOWN"},
        "calibration_gate": calibration or {"status": "UNKNOWN"},
        "targets": tasks,
        "target_isolation": {
            "score_derived_ou_btts_are_not_counted_as_standalone_evidence": True,
            "mom_requires_target_specific_artifact": True,
        },
        "evidence_hashes": evidence_hashes,
    }


def write_csv(snapshot: dict[str, Any], path: Path) -> None:
    fields = [
        "task",
        "status",
        "n",
        "blocks",
        "source_file",
        "logloss",
        "accuracy",
        "brier",
        "rps",
        "ece",
        "score_logloss",
        "exact_score_hit_rate",
        "top3_score_hit_rate",
        "top4_score_hit_rate",
        "home_goals_mae",
        "away_goals_mae",
        "total_goals_mae",
        "top1_accuracy",
        "top4_hit_rate",
        "mrr",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in snapshot["targets"]:
            row = {
                "task": item["task"],
                "status": item["status"],
                "n": item.get("n", 0),
                "blocks": item.get("blocks", 0),
                "source_file": item.get("source_file") or "",
            }
            row.update(item.get("metrics", {}))
            writer.writerow(row)


def write_markdown(snapshot: dict[str, Any], path: Path) -> None:
    lines = [
        "# Soccer Performance Snapshot",
        "",
        f"- Overall status: **{snapshot['status']}**",
        f"- PIT gate: **{snapshot['pit_gate']}**",
        f"- Adoption status: **{snapshot['adoption_status']}**",
        "",
        "| Target | Status | N | Blocks | Key metrics |",
        "|---|---|---:|---:|---|",
    ]
    for item in snapshot["targets"]:
        metrics = item.get("metrics", {})
        compact = ", ".join(
            f"{key}={value:.6f}" for key, value in metrics.items()
        ) or "-"
        lines.append(
            f"| {item['task']} | {item['status']} | {int(item.get('n', 0))} | "
            f"{int(item.get('blocks', 0))} | {compact} |"
        )
    lines.extend([
        "",
        "This report does not convert missing PIT evidence into performance claims.",
        "O/U and BTTS require target-specific evidence; fields embedded in score-model output are not relabeled as standalone validation.",
    ])
    if snapshot.get("pit_blocking_reasons"):
        lines.extend(["", "## Blocking reasons"])
        lines.extend(f"- {reason}" for reason in snapshot["pit_blocking_reasons"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--json-output", default="artifacts/performance_snapshot.json")
    parser.add_argument("--csv-output", default="artifacts/performance_snapshot.csv")
    parser.add_argument("--markdown-output", default="artifacts/performance_snapshot.md")
    args = parser.parse_args()

    root = Path(args.artifacts_dir)
    snapshot = build_snapshot(root)

    for raw in (args.json_output, args.csv_output, args.markdown_output):
        Path(raw).parent.mkdir(parents=True, exist_ok=True)

    Path(args.json_output).write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    write_csv(snapshot, Path(args.csv_output))
    write_markdown(snapshot, Path(args.markdown_output))

    print(json.dumps({
        "status": snapshot["status"],
        "pit_gate": snapshot["pit_gate"],
        "adoption_status": snapshot["adoption_status"],
        "targets": {
            item["task"]: item["status"] for item in snapshot["targets"]
        },
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
