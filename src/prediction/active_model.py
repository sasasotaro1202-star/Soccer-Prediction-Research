from __future__ import annotations

from pathlib import Path

CURRENT_PRODUCTION_DIR = Path("models/current")


def resolve_active_production_paths(bundle_path: str, registry_path: str) -> tuple[str, str]:
    """Resolve an adopted production bundle without losing the last valid model.

    Explicit caller-supplied paths remain intact (important for isolated tests and
    one-off runs). The durable fallback is used only for the standard production
    artifact locations when those artifacts are absent.
    """
    bundle = Path(bundle_path).expanduser()
    registry = Path(registry_path).expanduser()
    if bundle.is_file() and registry.is_file():
        return str(bundle.resolve()), str(registry.resolve())

    standard_bundle = Path("artifacts/production_model.pkl")
    standard_registry = Path("artifacts/model_registry.json")
    requested_standard = bundle == standard_bundle and registry == standard_registry
    if requested_standard:
        durable_bundle = CURRENT_PRODUCTION_DIR / "production_model.pkl"
        durable_registry = CURRENT_PRODUCTION_DIR / "model_registry.json"
        if durable_bundle.is_file() and durable_registry.is_file():
            return str(durable_bundle), str(durable_registry)

    # Preserve explicit paths so the normal loader can emit its own precise
    # fail-closed error (or a test can inject mocked loaders).
    return str(bundle), str(registry)


def resolve_best_available_paths() -> tuple[str, str, str]:
    """Resolve the safest currently available model: ADOPT first, then validated candidate."""
    candidates = [
        ("PRODUCTION_ADOPTED", Path("artifacts/production_model.pkl"), Path("artifacts/model_registry.json")),
        ("PRODUCTION_ADOPTED", CURRENT_PRODUCTION_DIR / "production_model.pkl", CURRENT_PRODUCTION_DIR / "model_registry.json"),
        ("VALIDATED_CANDIDATE", Path("artifacts/validated_candidate_model.pkl"), Path("artifacts/validated_candidate_registry.json")),
        ("VALIDATED_CANDIDATE", CURRENT_PRODUCTION_DIR / "validated_candidate_model.pkl", CURRENT_PRODUCTION_DIR / "validated_candidate_registry.json"),
    ]
    for mode, bundle, registry in candidates:
        if not bundle.is_file() or not registry.is_file():
            continue
        try:
            import json
            payload=json.loads(registry.read_text(encoding='utf-8'))
        except Exception:
            continue
        status=str(payload.get('adoption_status','')).upper()
        if mode == 'PRODUCTION_ADOPTED' and status == 'ADOPT':
            return str(bundle), str(registry), mode
        if mode == 'VALIDATED_CANDIDATE' and status == 'VALIDATED_CANDIDATE':
            return str(bundle), str(registry), mode
    raise RuntimeError("No safe production or validated-candidate model is currently available")
