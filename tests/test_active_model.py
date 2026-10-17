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

def test_resolver_best_available_can_use_remote_fallback(monkeypatch, tmp_path):
    import src.prediction.active_model as active_model

    monkeypatch.chdir(tmp_path)
    root = tmp_path / "remote"

    def fake_fetch(repository):
        assert repository == "owner/repo"
        root.mkdir()
        (root / "production_model.pkl").write_bytes(b"p")
        (root / "model_registry.json").write_text("{}", encoding="utf-8")
        return root

    monkeypatch.setattr(active_model, "resolve_best_available_paths", lambda: (_ for _ in ()).throw(RuntimeError("none")))
    monkeypatch.setattr("src.prediction.current_production.fetch_current_production", fake_fetch)

    bundle, registry, mode = active_model.resolve_best_available_paths_with_remote("owner/repo")
    assert bundle == str(root / "production_model.pkl")
    assert registry == str(root / "model_registry.json")
    assert mode == "PRODUCTION_ADOPTED"
