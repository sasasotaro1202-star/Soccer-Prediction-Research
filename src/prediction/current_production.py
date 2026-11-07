from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

CURRENT_RELEASE_TAG = "production-current"
CURRENT_RELEASE_ASSET = "production-current.zip"
REQUIRED_FILES = (
    "production_model.pkl",
    "production_model.json",
    "model_registry.json",
    "production_provenance.json",
    "calibration_gate.json",
    "adoption_decision.json",
    "production_contract.json",
)


def _repo(repository: str) -> str:
    value = repository.strip().strip("/")
    if not value or value.count("/") != 1:
        raise ValueError("repository must be owner/name")
    return value


def manifest_url(repository: str) -> str:
    return f"https://raw.githubusercontent.com/{_repo(repository)}/main/production/current.json"


def release_asset_url(repository: str) -> str:
    return f"https://github.com/{_repo(repository)}/releases/download/{CURRENT_RELEASE_TAG}/{CURRENT_RELEASE_ASSET}"


def load_current_manifest(repository: str, timeout: int = 20) -> dict:
    request = Request(manifest_url(repository), headers={"User-Agent": "soccer-production-reader"})
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "READY":
        raise RuntimeError("No verified current production bundle is published")
    if str(payload.get("adoption_status", "")).upper() != "ADOPT":
        raise RuntimeError("Current production manifest is not ADOPT")
    return payload


def _safe_extract(zf: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for info in zf.infolist():
        target = (destination / info.filename).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError(f"Unsafe production archive path: {info.filename}")
    zf.extractall(destination)


def fetch_current_production(repository: str, output_dir: str = "artifacts/current_production", timeout: int = 30) -> Path:
    manifest = load_current_manifest(repository, timeout=timeout)
    request = Request(release_asset_url(repository), headers={"User-Agent": "soccer-production-reader"})
    with urlopen(request, timeout=timeout) as response:
        archive = response.read()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        names = {n.rstrip("/") for n in zf.namelist()}
        missing = [name for name in REQUIRED_FILES if name not in names]
        if missing:
            raise RuntimeError(f"Current production release missing files: {missing}")
        _safe_extract(zf, root)
    contract = json.loads((root / "production_contract.json").read_text(encoding="utf-8"))
    adoption = json.loads((root / "adoption_decision.json").read_text(encoding="utf-8"))
    registry = json.loads((root / "model_registry.json").read_text(encoding="utf-8"))
    calibration = json.loads((root / "calibration_gate.json").read_text(encoding="utf-8"))
    if contract.get("production_contract_passed") is not True:
        raise RuntimeError("Published production bundle contract is not PASS")
    if str(adoption.get("status", "")).upper() != "ADOPT" or str(registry.get("adoption_status", "")).upper() != "ADOPT":
        raise RuntimeError("Published production bundle adoption is not ADOPT")
    if calibration.get("status") != "PASS":
        raise RuntimeError("Published production bundle calibration is not PASS")
    expected_hash = manifest.get("production_model_sha256") or manifest.get("bundle_sha256")
    if expected_hash:
        actual_hash = hashlib.sha256((root / "production_model.pkl").read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError("Published production model SHA256 mismatch")
    return root
