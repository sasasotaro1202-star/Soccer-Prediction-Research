"""Research-only post-outcome experience ledger for soccer predictions.

The ledger closes the loop only after an outcome is explicitly confirmed.
Prediction snapshots are append-only and never rewritten. Outcome records must
carry explicit availability evidence strictly after the prediction cutoff.

No model, production registry, or promotion gate is modified here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROBABILITY_KEYS = ("home_win_pct", "draw_pct", "away_win_pct")
OUTCOME_LABELS = ("HOME_WIN", "DRAW", "AWAY_WIN")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(value: Any) -> datetime:
    if value is None:
        raise ValueError("timestamp is required")
    raw = str(value).strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include timezone")
    return parsed.astimezone(timezone.utc)


def _finite_probability(value: Any) -> float:
    x = float(value)
    if not math.isfinite(x) or x < 0.0 or x > 100.0:
        raise ValueError("probability must be finite and in [0, 100]")
    return x / 100.0


def prediction_id(row: dict[str, Any]) -> str:
    required = ("match_id", "prediction_cutoff_utc", "git_commit")
    if any(not row.get(k) for k in required):
        raise ValueError("prediction_id requires match_id, prediction_cutoff_utc and git_commit")
    raw = "|".join(str(row[k]) for k in required)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_no} is not an object")
        rows.append(value)
    return rows


def archive_predictions(
    input_json: str | Path,
    output_jsonl: str | Path,
    *,
    run_id: str | None = None,
) -> dict[str, int]:
    """Archive only explicitly executed, pre-match prediction snapshots."""
    payload = _read_json(input_json)
    if payload.get("execution_status") != "EXECUTED":
        return {"archived": 0, "skipped": len(payload.get("predictions", []) or [])}

    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("predictions must be a list")

    out = Path(output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = {str(row.get("prediction_id")): row for row in _read_jsonl(out) if row.get("prediction_id")}

    archived = skipped = 0
    for raw in predictions:
        if not isinstance(raw, dict):
            raise ValueError("prediction row must be an object")
        for key in ("match_id", "prediction_cutoff_utc", "prediction_generated_at", "home", "away", "git_commit"):
            if not raw.get(key):
                raise ValueError(f"prediction row missing {key}")

        cutoff = _parse_ts(raw["prediction_cutoff_utc"])
        generated = _parse_ts(raw["prediction_generated_at"])
        kickoff = _parse_ts(raw["kickoff_utc"]) if raw.get("kickoff_utc") else None
        if kickoff is not None and cutoff >= kickoff:
            skipped += 1
            continue
        if generated < cutoff:
            raise ValueError("prediction_generated_at cannot precede prediction_cutoff_utc")

        probabilities = {key: _finite_probability(raw.get(key)) for key in PROBABILITY_KEYS}
        total = sum(probabilities.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError("prediction probabilities must sum to 100 percent")

        record = dict(raw)
        record["prediction_id"] = prediction_id(record)
        record["source_run_id"] = str(run_id) if run_id is not None else None
        record["archived_at_utc"] = _utc_now()
        record["prediction_pit_validated"] = True
        key = record["prediction_id"]
        if key in existing:
            skipped += 1
        else:
            existing[key] = record
            with out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            archived += 1

    return {"archived": archived, "skipped": skipped}


def _multiclass_ece(probabilities: list[list[float]], y_true: list[int], bins: int = 10) -> float:
    if len(probabilities) != len(y_true) or not y_true:
        raise ValueError("invalid ECE inputs")
    confidence = [max(row) for row in probabilities]
    predicted = [max(range(len(row)), key=row.__getitem__) for row in probabilities]
    edges = [i / bins for i in range(bins + 1)]
    total = len(y_true)
    ece = 0.0
    for i in range(bins):
        upper = edges[i + 1]
        idx = [
            j for j, conf in enumerate(confidence)
            if conf >= edges[i] and (conf < upper if i < bins - 1 else conf <= upper)
        ]
        if idx:
            acc = sum(int(predicted[j] == y_true[j]) for j in idx) / len(idx)
            mean_conf = sum(confidence[j] for j in idx) / len(idx)
            ece += len(idx) / total * abs(mean_conf - acc)
    return float(ece)


def reconcile_experience(
    predictions_jsonl: str | Path,
    outcomes_jsonl: str | Path,
    ledger_jsonl: str | Path,
) -> dict[str, Any]:
    """Reconcile matured outcomes; fail closed on missing/early outcome evidence."""
    predictions = _read_jsonl(predictions_jsonl)
    outcomes = _read_jsonl(outcomes_jsonl)
    by_match = {}
    for outcome in outcomes:
        match_id = str(outcome.get("match_id") or "")
        if not match_id:
            raise ValueError("outcome missing match_id")
        if match_id in by_match:
            raise ValueError(f"duplicate outcome for {match_id}")
        by_match[match_id] = outcome

    ledger_path = Path(ledger_jsonl)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {str(row.get("prediction_id")): row for row in _read_jsonl(ledger_path) if row.get("prediction_id")}

    new_rows: list[dict[str, Any]] = []
    blocked = 0
    for pred in predictions:
        match_id = str(pred.get("match_id") or "")
        if not match_id or str(pred.get("prediction_id") or "") == "":
            raise ValueError("prediction missing match_id or prediction_id")
        outcome = by_match.get(match_id)
        if outcome is None:
            blocked += 1
            continue

        cutoff = _parse_ts(pred["prediction_cutoff_utc"])
        available = _parse_ts(outcome.get("available_at_utc"))
        if available <= cutoff:
            raise ValueError(f"outcome availability is not after prediction cutoff: {match_id}")
        if outcome.get("status") != "FINAL":
            blocked += 1
            continue
        actual = str(outcome.get("actual_outcome") or "")
        if actual not in OUTCOME_LABELS:
            raise ValueError(f"invalid actual_outcome: {match_id}")

        probs = [_finite_probability(pred.get(key)) for key in PROBABILITY_KEYS]
        actual_idx = OUTCOME_LABELS.index(actual)
        predicted_idx = max(range(3), key=probs.__getitem__)
        logloss = -math.log(max(probs[actual_idx], 1e-12))
        brier = sum((probs[i] - (1.0 if i == actual_idx else 0.0)) ** 2 for i in range(3))
        record = dict(pred)
        record.update({
            "actual_outcome": actual,
            "predicted_outcome": OUTCOME_LABELS[predicted_idx],
            "outcome_correct": int(predicted_idx == actual_idx),
            "logloss": float(logloss),
            "brier": float(brier),
            "outcome_available_at_utc": outcome["available_at_utc"],
            "experience_available_at_utc": _utc_now(),
            "outcome_pit_check": "PASS",
        })
        key = str(pred["prediction_id"])
        if key not in existing:
            existing[key] = record
            new_rows.append(record)

    if new_rows:
        with ledger_path.open("a", encoding="utf-8") as fh:
            for row in new_rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    all_rows = list(existing.values())
    if not all_rows:
        return {
            "status": "NO_MATURED_OUTCOMES",
            "prediction_rows": len(predictions),
            "matured_rows": 0,
            "blocked_rows": blocked,
        }

    probs_matrix = [
        [_finite_probability(row.get(key)) for key in PROBABILITY_KEYS]
        for row in all_rows
    ]
    y = [OUTCOME_LABELS.index(str(row["actual_outcome"])) for row in all_rows]
    correct = [int(row["outcome_correct"]) for row in all_rows]
    summary = {
        "status": "UPDATED",
        "prediction_rows": len(predictions),
        "matured_rows": len(all_rows),
        "newly_matured_rows": len(new_rows),
        "blocked_rows": blocked,
        "accuracy": sum(correct) / len(correct),
        "logloss": sum(float(row["logloss"]) for row in all_rows) / len(all_rows),
        "brier": sum(float(row["brier"]) for row in all_rows) / len(all_rows),
        "ece": _multiclass_ece(probs_matrix, y),
        "research_only": True,
        "production_changed": False,
        "teacher_information_available_only_after_outcome": True,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    archive = sub.add_parser("archive")
    archive.add_argument("--input", required=True)
    archive.add_argument("--output", required=True)
    archive.add_argument("--run-id")

    reconcile = sub.add_parser("reconcile")
    reconcile.add_argument("--predictions", required=True)
    reconcile.add_argument("--outcomes", required=True)
    reconcile.add_argument("--ledger", required=True)

    args = parser.parse_args()
    if args.command == "archive":
        result = archive_predictions(args.input, args.output, run_id=args.run_id)
    else:
        result = reconcile_experience(args.predictions, args.outcomes, args.ledger)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
