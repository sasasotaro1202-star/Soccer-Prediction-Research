from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

CURRENT_RELEASE_TAG = 'production-current'
CURRENT_RELEASE_ASSET = 'production-current.zip'
REQUIRED_FILES = (
    'production_model.pkl',
    'production_model.json',
    'model_registry.json',
    'production_provenance.json',
    'calibration_gate.json',
    'adoption_decision.json',
    'production_contract.json',
)


def release_asset_url(repository: str) -> str:
    owner_repo = repository.strip().strip('/')
    if not owner_repo or owner_repo.count('/') != 1:
        raise ValueError('repository must be owner/name')
    return f'https://github.com/{owner_repo}/releases/download/{CURRENT_RELEASE_TAG}/{CURRENT_RELEASE_ASSET}'


def manifest_url(repository: str) -> str:
    owner_repo = repository.strip().strip('/')
    if not owner_repo or owner_repo.count('/') != 1:
        raise ValueError('repository must be owner/name')
    return f'https://raw.githubusercontent.com/{owner_repo}/main/production/current.json'


def load_current_manifest(repository: str, timeout: int = 20) -> dict:
    request = Request(manifest_url(repository), headers={'User-Agent': 'soccer-production-reader'})
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode('utf-8'))
    if not isinstance(payload, dict) or payload.get('status') != 'READY':
        raise RuntimeError('No verified current production bundle is published')
    if str(payload.get('adoption_status', '')).upper() != 'ADOPT':
        raise RuntimeError('Current production manifest is not ADOPT')
    return payload


def fetch_current_production(repository: str, output_dir: str = 'artifacts/current_production', timeout: int = 30) -> Path:
    manifest = load_current_manifest(repository, timeout=timeout)
    request = Request(release_asset_url(repository), headers={'User-Agent': 'soccer-production-reader'})
    with urlopen(request, timeout=timeout) as response:
        archive = response.read()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        names = set(zf.namelist())
        missing = [name for name in REQUIRED_FILES if name not in names]
        if missing:
            raise RuntimeError(f'Current production release missing files: {missing}')
        zf.extractall(root)
    contract = json.loads((root / 'production_contract.json').read_text(encoding='utf-8'))
    adoption = json.loads((root / 'adoption_decision.json').read_text(encoding='utf-8'))
    registry = json.loads((root / 'model_registry.json').read_text(encoding='utf-8'))
    calibration = json.loads((root / 'calibration_gate.json').read_text(encoding='utf-8'))
    if contract.get('production_contract_passed') is not True:
        raise RuntimeError('Published production bundle contract is not PASS')
    if str(adoption.get('status', '')).upper() != 'ADOPT' or str(registry.get('adoption_status', '')).upper() != 'ADOPT':
        raise RuntimeError('Published production bundle adoption is not ADOPT')
    if calibration.get('status') != 'PASS':
        raise RuntimeError('Published production bundle calibration gate is not PASS')
    return root