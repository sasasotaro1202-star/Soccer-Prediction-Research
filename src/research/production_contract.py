from __future__ import annotations

"""Fail-closed production research contract."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from typing import Any

REQUIRED_GATES = ("data", "schema", "leakage", "features", "training", "backtest", "oos", "prediction", "sanity", "artifact")
REQUIRED_ARTIFACTS = (
    "oos_metrics.csv",
    "model_selection.csv",
    "development_oos_metrics.csv",
    "locked_oos_metrics.csv",
    "candidate_lock.json",
    "adoption_decision.json",
)
PROVENANCE_ARTIFACTS = (
    "production_model.pkl",
    "production_model.json",
    "model_registry.json",
    "predictions.csv",
    "future_fixtures.csv",
)

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


def _file_nonempty(root: Path, name: str) -> bool:
    return (root / name).is_file() and (root / name).stat().st_size > 0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_hash(value: Any) -> str | None:
    if value is None:
        return None
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(raw).hexdigest()


def _write_provenance(root: Path) -> None:
    """Write immutable-in-run hashes for deployable and prediction artifacts."""
    files: dict[str, dict[str, Any]] = {}
    for name in PROVENANCE_ARTIFACTS:
        path = root / name
        if path.is_file():
            files[name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    model_json = _read_json(root / "production_model.json")
    registry = _read_json(root / "model_registry.json")
    feature_schema = model_json.get("feature_cols")
    if feature_schema is None:
        feature_schema = registry.get("feature_cols")
    payload = {
        "provenance_schema_version": 1,
        "files": files,
        "model_version": model_json.get("model_version") or registry.get("model_version"),
        "feature_schema_sha256": _canonical_hash(feature_schema),
        "registry_model_version": registry.get("model_version"),
        "production_model_json_version": model_json.get("model_version"),
    }
    (root / "production_provenance.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )


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
    if not adoption:
        failures.append("adoption_missing")
    else:
        adoption_status = str(adoption.get("status", "")).upper()
        if adoption_status not in {"ADOPT", "CHAMPION", "ADOPTED"}:
            failures.append(f"adoption:{adoption.get('status')}")
        if adoption.get("oos_claimed") is not True:
            failures.append("adoption_oos_claim")
        stability = adoption.get("stability")
        if not isinstance(stability, dict) or stability.get("status") not in {"PASS"}:
            failures.append("adoption_stability")
        if adoption_status == "ADOPT":
            if not _file_nonempty(root, "production_model.pkl"):
                failures.append("artifact:production_model.pkl")
            registry = _read_json(root / "model_registry.json")
            if registry.get("adoption_status") != "ADOPT":
                failures.append("artifact:model_registry.json")
            model_json = _read_json(root / "production_model.json")
            if model_json.get("adoption_status") != "ADOPT":
                failures.append("artifact:production_model.json")
            registry_version = registry.get("model_version")
            model_json_version = model_json.get("model_version")
            if registry_version is not None and model_json_version is not None and str(registry_version) != str(model_json_version):
                failures.append("model_version_provenance_mismatch")
            registry_features = registry.get("feature_cols")
            model_features = model_json.get("feature_cols")
            if registry_features is not None and model_features is not None:
                if _canonical_hash(registry_features) != _canonical_hash(model_features):
                    failures.append("feature_schema_provenance_mismatch")
    candidate_lock = _read_json(root / "candidate_lock.json")
    if candidate_lock:
        if candidate_lock.get("locked_oos_untouched") is not True: failures.append("candidate_lock_integrity")
        if int(candidate_lock.get("locked_oos_blocks", 0)) < 2: failures.append("locked_oos_blocks")
    else:
        failures.append("candidate_lock_missing")
    return GateResult(not failures, tuple(dict.fromkeys(failures)))


def write_contract_result(artifacts_dir: str = "artifacts") -> GateResult:
    root = Path(artifacts_dir); root.mkdir(parents=True, exist_ok=True)
    result = evaluate_production_contract(artifacts_dir)
    _write_provenance(root)
    (root / "production_contract.json").write_text(
        json.dumps({
            "production_contract_passed": result.passed,
            "failures": list(result.failures),
            "required_gates": list(REQUIRED_GATES),
            "required_artifacts": list(REQUIRED_ARTIFACTS),
            "provenance_artifact": "production_provenance.json",
            "fail_closed": True,
        }, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    result = write_contract_result(); print(json.dumps({"passed": result.passed, "failures": list(result.failures)}, ensure_ascii=False)); raise SystemExit(0 if result.passed else 1)
