from __future__ import annotations

import csv
import json
from pathlib import Path

from src.research.performance_snapshot import build_snapshot, write_csv


def _write_json(root: Path, name: str, payload: dict):
    (root / name).write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(root: Path, name: str, rows: list[dict]):
    path = root / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_blocked_pit_never_becomes_evaluated(tmp_path):
    _write_json(tmp_path, "completion_gate.json", {
        "full_gate_passed": False,
        "blocking_reasons": ["pit_publication_time_gate=false"],
    })
    _write_json(tmp_path, "audit_gate.json", {
        "full_gate_passed": False,
        "blocking_reasons": ["publication_time_missing"],
    })
    snapshot = build_snapshot(tmp_path)
    assert snapshot["status"] == "BLOCKED"
    assert snapshot["pit_gate"] == "BLOCKED"
    assert all(item["status"] == "BLOCKED" for item in snapshot["targets"])


def test_weighted_locked_oos_metrics_are_aggregated(tmp_path):
    _write_json(tmp_path, "completion_gate.json", {"full_gate_passed": True})
    _write_json(tmp_path, "audit_gate.json", {"full_gate_passed": True})
    _write_csv(tmp_path, "locked_oos_metrics.csv", [
        {"n": "100", "logloss": "0.60", "accuracy": "0.65", "brier": "0.20", "ece": "0.04"},
        {"n": "300", "logloss": "0.40", "accuracy": "0.70", "brier": "0.16", "ece": "0.02"},
    ])
    snapshot = build_snapshot(tmp_path)
    one_x_two = next(item for item in snapshot["targets"] if item["task"] == "1X2")
    assert one_x_two["status"] == "EVALUATED"
    assert one_x_two["n"] == 400
    assert abs(one_x_two["metrics"]["logloss"] - 0.45) < 1e-12
    assert abs(one_x_two["metrics"]["accuracy"] - 0.6875) < 1e-12


def test_score_derived_ou_btts_do_not_count_as_standalone_targets(tmp_path):
    _write_json(tmp_path, "completion_gate.json", {"full_gate_passed": True})
    _write_json(tmp_path, "audit_gate.json", {"full_gate_passed": True})
    _write_csv(tmp_path, "score_locked_oos_metrics.csv", [
        {
            "n": "500",
            "score_logloss": "2.1",
            "top3_score_hit_rate": "0.42",
            "over_2_5_logloss": "0.61",
            "over_2_5_brier": "0.21",
            "btts_logloss": "0.64",
            "btts_brier": "0.22",
        },
    ])
    snapshot = build_snapshot(tmp_path)
    score = next(item for item in snapshot["targets"] if item["task"] == "Score")
    ou = next(item for item in snapshot["targets"] if item["task"] == "O/U")
    btts = next(item for item in snapshot["targets"] if item["task"] == "BTTS")
    assert score["status"] == "EVALUATED"
    assert ou["status"] == "UNVERIFIED"
    assert btts["status"] == "UNVERIFIED"


def test_csv_output_preserves_target_rows(tmp_path):
    _write_json(tmp_path, "completion_gate.json", {"full_gate_passed": True})
    _write_json(tmp_path, "audit_gate.json", {"full_gate_passed": True})
    snapshot = build_snapshot(tmp_path)
    out = tmp_path / "out.csv"
    write_csv(snapshot, out)
    rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
    assert {row["task"] for row in rows} == {"1X2", "Score", "O/U", "BTTS", "MOM"}
