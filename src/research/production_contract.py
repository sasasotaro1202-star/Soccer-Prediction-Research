from __future__ import annotations

"""Fail-closed production research contract."""

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

REQUIRED_GATES = ("data", "schema", "leakage", "features", "training", "backtest", "oos", "prediction", "sanity", "artifact")
REQUIRED_ARTIFACTS = ("oos_metrics.csv", "model_selection.csv", "adoption_decision.json")

@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: tuple[str, ...]

def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists(): return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}

def _bool_artifact(root: Path, name: str, key: str = "passed") -> bool:
    return _read_json(root / name).get(key) is True

def _csv_nonempty(root: Path, name: str) -> bool:
    path = root / name
    if not path.exists() or path.stat().st_size == 0: return False
    try: return len(path.read_text(encoding="utf-8").splitlines()) >= 2
    except Exception: return False

def evaluate_production_contract(artifacts_dir: str = "artifacts") -> GateResult:
    root = Path(artifacts_dir); failures: list[str] = []
    completion = _read_json(root / "completion_gate.json")
    if completion.get("full_gate_passed") is not True: failures.append("completion_gate")
    if not _bool_artifact(root, "test_status.json"): failures.append("tests")
    if not _bool_artifact(root, "audit_status.json"): failures.append("audit_execution")
    audit_gate = _read_json(root / "audit_gate.json")
    if audit_gate.get("full_gate_passed") is not True: failures.append("audit_gate")
    status = _read_json(root / "run_status.json")
    if status.get("oos_claimed") is not True: failures.append("oos_claim")
    gate_map = status.get("gates")
    if not isinstance(gate_map, dict): failures.append("gate_map_missing")
    else:
        for name in REQUIRED_GATES:
            if gate_map.get(name) is not True: failures.append(name)
    for filename in REQUIRED_ARTIFACTS:
        if filename.endswith(".csv"):
            if not _csv_nonempty(root, filename): failures.append(f"artifact:{filename}")
        elif not (root / filename).exists(): failures.append(f"artifact:{filename}")
    if str(status.get("status", "")).upper() in {"BLOCKED", "FAIL", "FAILED", "DEGRADED"}: failures.append(f"run_status:{status.get('status')}")
    if completion.get("pit_publication_time_gate") is not True: failures.append("pit_publication_time_gate")
    adoption = _read_json(root / "adoption_decision.json")
    if not adoption: failures.append("adoption_missing")
    elif str(adoption.get("status", "")).upper() in {"REJECT", "BLOCKED", "FAIL", "DEGRADED", "NO_CHAMPION"}: failures.append(f"adoption:{adoption.get('status')}")
    return GateResult(not failures, tuple(dict.fromkeys(failures)))

def write_contract_result(artifacts_dir: str = "artifacts") -> GateResult:
    root = Path(artifacts_dir); root.mkdir(parents=True, exist_ok=True)
    result = evaluate_production_contract(artifacts_dir)
    (root / "production_contract.json").write_text(json.dumps({"production_contract_passed": result.passed, "failures": list(result.failures), "required_gates": list(REQUIRED_GATES), "required_artifacts": list(REQUIRED_ARTIFACTS), "fail_closed": True}, indent=2, ensure_ascii=False), encoding="utf-8")
    return result

if __name__ == "__main__":
    result = write_contract_result(); print(json.dumps({"passed": result.passed, "failures": list(result.failures)}, ensure_ascii=False)); raise SystemExit(0 if result.passed else 1)
