from pathlib import Path

from src.prediction.active_model import resolve_active_production_paths


def test_resolver_prefers_run_local_bundle(tmp_path, monkeypatch):
    bundle=tmp_path/'production_model.pkl'; registry=tmp_path/'model_registry.json'
    bundle.write_bytes(b'local'); registry.write_text('{}', encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    resolved=resolve_active_production_paths('artifacts/production_model.pkl','artifacts/model_registry.json')
    assert resolved==(str(bundle),str(registry))


def test_resolver_uses_durable_current_bundle(tmp_path, monkeypatch):
    durable=tmp_path/'models'/'current'; durable.mkdir(parents=True)
    bundle=durable/'production_model.pkl'; registry=durable/'model_registry.json'
    bundle.write_bytes(b'durable'); registry.write_text('{}', encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    resolved=resolve_active_production_paths('artifacts/production_model.pkl','artifacts/model_registry.json')
    assert resolved==(str(bundle),str(registry))
