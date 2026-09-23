from __future__ import annotations

from pathlib import Path

CURRENT_PRODUCTION_DIR = Path("models/current")


def resolve_active_production_paths(bundle_path: str, registry_path: str) -> tuple[str, str]:
    """Use the run-local adopted bundle; otherwise use the durable last-ADOPT bundle."""
    bundle = Path(bundle_path)
    registry = Path(registry_path)
    if bundle.is_file() and registry.is_file():
        return str(bundle), str(registry)
    durable_bundle = CURRENT_PRODUCTION_DIR / "production_model.pkl"
    durable_registry = CURRENT_PRODUCTION_DIR / "model_registry.json"
    if durable_bundle.is_file() and durable_registry.is_file():
        return str(durable_bundle), str(durable_registry)
    raise RuntimeError(
        "No current production model is available: neither a run-local ADOPT bundle "
        "nor the durable last-ADOPT production bundle exists."
    )
