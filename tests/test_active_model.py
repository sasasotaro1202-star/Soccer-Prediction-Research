from pathlib import Path

from src.prediction.active_model import resolve_active_production_paths


def test_resolver_prefers_run_local_bundle(tmp_path, monkeypatch):
    bundle = tmp_path / "production_model.pkl"
    registry = tmp_path / "model_registry.json"
    bundle.write_bytes(b"local")
    registry.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    local = tmp_path / "artifacts"
    local.mkdir()
    (local / "production_model.pkl").write_bytes(b"local")
    (local / "model_registry.json").write_text("{}", encoding="utf-8")
    resolved = resolve_active_production_paths("artifacts/production_model.pkl", "artifacts/model_registry.json")
    assert resolved == (str(local / "production_model.pkl"), str(local / "model_registry.json"))


def test_resolver_uses_durable_current_bundle(tmp_path, monkeypatch):
    durable = tmp_path / "models" / "current"
    durable.mkdir(parents=True)
    bundle = durable / "production_model.pkl"
    registry = durable / "model_registry.json"
    bundle.write_bytes(b"durable")
    registry.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    resolved = resolve_active_production_paths("artifacts/production_model.pkl", "artifacts/model_registry.json")
    assert resolved == (str(bundle.resolve()), str(registry.resolve()))

def test_remote_fallback_returns_safe_production_mode(monkeypatch):
    from src.prediction import active_model

    monkeypatch.setattr(
        active_model,
        "fetch_current_production",
        lambda repository: __import__("pathlib").Path("artifacts/current_production"),
        raising=False,
    )
