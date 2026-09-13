from __future__ import annotations

"""Fail-closed production research contract.

This module contains policy, not model logic. A successful Python process or
GitHub job is never sufficient evidence of a valid research/production run.
"""

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any


REQUIRED_GATES = (
    "data",
    "schema",
    "leakage",
    "features",
    "training",
    "backtest",
    "oos",
    "prediction",
    "sanity",
    "artifact",
)


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: tuple[str, ...]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _bool_artifact(root: Path, name: str, key: str = "passed") -> bool:
    payload = _read_json(root / name)
    return payload.get(key) is True


def evaluate_production_contract(artifacts_dir: str = "artifacts") -> GateResult:
    """Evaluate explicit research gates; missing/unknown evidence fails closed."""
    root = Path(artifacts_dir)
    failures: list[str] = []

    completion = _read_json(root / "completion_gate.json")
    if completion.get("full_gate_passed") is not True:
        failures.append("completion_gate")

    # Preflight evidence must exist independently of run_status.json.
    if not _bool_artifact(root, "test_status.json"):
        failures.append("tests")
    if not _bool_artifact(root, "audit_status.json"):
        failures.append("audit_execution")

    audit_gate = _read_json(root / "audit_gate.json")
    if audit_gate.get("full_gate_passed") is not True:
        failures.append("audit_gate")

    status = _read_json(root / "run_status.json")
    if status.get("oos_claimed") is not True:
        failures.append("oos_claim")

    # Detailed gate maps are authoritative when present.
    gate_map = status.get("gates")
    if isinstance(gate_map, dict):
        for name in REQUIRED_GATES:
            if gate_map.get(name) is not True:
                failures.append(name)

    # Research artifacts required for a real OOS claim.
    for filename in ("oos_metrics.csv", "model_selection.csv", "adoption_decision.json"):
        if not (root / filename).exists():
            failures.append(f"artifact:{filename}")

    # A blocked/degraded research state can never be promoted by CI success.
    if str(status.get("status", "")).upper() in {"BLOCKED", "FAIL", "FAILED", "DEGRADED"}:
        failures.append(f"run_status:{status.get('status')}")

    return GateResult(passed=not failures, failures=tuple(dict.fromkeys(failures)))


def write_contract_result(artifacts_dir: str = "artifacts") -> GateResult:
    root = Path(artifacts_dir)
    root.mkdir(parents=True, exist_ok=True)
    result = evaluate_production_contract(artifacts_dir)
    payload = {
        "production_contract_passed": result.passed,
        "failures": list(result.failures),
        "required_gates": list(REQUIRED_GATES),
        "fail_closed": True,
    }
    (root / "production_contract.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    result = write_contract_result()
    print(json.dumps({"passed": result.passed, "failures": list(result.failures)}, ensure_ascii=False))
    raise SystemExit(0 if result.passed else 1)
