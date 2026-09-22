from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.baselines import candidates
from src.models.dixon_coles import fit_dixon_coles_model
from src.prediction.secondary_outputs import fit_recency_score_rate_model, fit_score_rate_model, fit_time_decay_score_rate_model


def _temperature_transform(proba: np.ndarray, temperature: float) -> np.ndarray:
    p = np.clip(np.asarray(proba, dtype=float), 1e-9, 1.0)
    logits = np.log(p) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    return out / out.sum(axis=1, keepdims=True)


def train_and_save_bundle(
    feats: pd.DataFrame,
    feature_cols: list[str],
    selection: dict[str, Any],
    output_path: str,
    model_version: str,
    data_snapshot_id: str,
    score_selection: dict[str, Any] | None = None,
    score_locked_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train the already-locked production ensemble on all data available after evaluation.

    Model/weight/calibration choices come only from the prior chronological validation
    record. The locked OOS outcomes are therefore never used to choose hyperparameters;
    after the evaluation is finalized they may be included in the final production fit.
    """
    if not feature_cols or any(not isinstance(c, str) or not c for c in feature_cols):
        raise ValueError("Production bundle requires a non-empty feature column list")
    if not model_version or not str(model_version).strip():
        raise ValueError("Production bundle requires a non-empty model_version")
    if not data_snapshot_id or not str(data_snapshot_id).strip():
        raise ValueError("Production bundle requires a non-empty data_snapshot_id")
    missing_features = [c for c in feature_cols if c not in feats.columns]
    if missing_features:
        raise ValueError(f"Training data missing model features: {missing_features[:10]}")
    if "pit_verified" not in feats.columns or "target" not in feats.columns:
        raise ValueError("Training data must contain pit_verified and target")
    d = feats[feats["pit_verified"] == True].dropna(subset=["target"]).copy()
    if d.empty:
        raise ValueError("Cannot build production bundle from empty PIT-verified data")
    weights = {str(k): float(v) for k, v in (selection.get("weights") or {}).items()}
    if not weights or any(not np.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("Locked selection contains invalid ensemble weights")
    total = sum(weights.values())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid ensemble weights")
    weights = {k: v / total for k, v in weights.items()}
    temperature = float(selection.get("temperature", 1.0) or 1.0)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid calibration temperature")

    fitted = {}
    available = candidates(random_state=42)
    for name, weight in weights.items():
        if name not in available:
            raise ValueError(f"Locked selection references unknown model: {name}")
        model = available[name]
        model.fit(d[feature_cols], d.target.astype(int))
        fitted[name] = model

    has_score_columns = {"home_team", "away_team", "home_goals", "away_goals"}.issubset(d.columns)
    score_selection = score_selection if isinstance(score_selection, dict) else {}
    score_locked_gate = score_locked_gate if isinstance(score_locked_gate, dict) else {}
    selected_score_method = "primary"
    requested_score_method = str(score_selection.get("selected_method", "primary"))
    verified_score_method = str(score_locked_gate.get("selected_method", "primary"))
    if (
        score_locked_gate.get("status") == "PASS"
        and requested_score_method in {"recency", "time_decay", "dixon_coles"}
        and verified_score_method == requested_score_method
    ):
        selected_score_method = requested_score_method

    score_model = None
    if has_score_columns:
        if selected_score_method == "recency":
            score_model = fit_recency_score_rate_model(d)
        elif selected_score_method == "time_decay":
            score_model = fit_time_decay_score_rate_model(d)
        elif selected_score_method == "dixon_coles":
            score_model = fit_dixon_coles_model(d)
        else:
            score_model = fit_score_rate_model(d)

    bundle = {
        "schema_version": 2 if score_model is not None else 1,
        "model_version": model_version,
        "data_snapshot_id": data_snapshot_id,
        "feature_cols": list(feature_cols),
        "weights": weights,
        "temperature": temperature,
        "models": fitted,
        "fit_rows": int(len(d)),
        "fit_end": str(d["kickoff_utc"].max()),
        "selection_source": "chronological_validation_locked_before_final_fit",
        "score_method": selected_score_method,
        "score_selection": score_selection,
        "score_locked_verification": score_locked_gate,
        **({"score_model": score_model} if score_model is not None else {}),
        "mom_model": {"status": "UPSTREAM_PLAYER_MODEL_REQUIRED", "output_top_k": 4},
    }
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(p)
    return {
        "path": str(p),
        "model_version": model_version,
        "data_snapshot_id": data_snapshot_id,
        "fit_rows": int(len(d)),
        "fit_end": str(d["kickoff_utc"].max()),
        "feature_count": len(feature_cols),
        "weights": weights,
        "temperature": temperature,
        "score_method": selected_score_method,
    }


def load_bundle(path: str = "artifacts/production_model.pkl") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Production model bundle missing: {p}")
    try:
        with p.open("rb") as fh:
            bundle = pickle.load(fh)
    except Exception as exc:
        raise RuntimeError(f"Production model bundle is unreadable: {type(exc).__name__}: {exc}") from exc
    if not isinstance(bundle, dict):
        raise RuntimeError("Production model bundle root must be a dictionary")
    required = {"schema_version", "model_version", "data_snapshot_id", "feature_cols", "weights", "temperature", "models"}
    missing = sorted(required - set(bundle))
    if missing:
        raise RuntimeError(f"Production model bundle missing fields: {missing}")
    if bundle["schema_version"] not in {1, 2}:
        raise RuntimeError(f"Unsupported production model bundle schema: {bundle['schema_version']}")
    if not isinstance(bundle["model_version"], str) or not bundle["model_version"].strip():
        raise RuntimeError("Production model bundle model_version is empty")
    if not isinstance(bundle["data_snapshot_id"], str) or not bundle["data_snapshot_id"].strip():
        raise RuntimeError("Production model bundle data_snapshot_id is empty")
    feature_cols = bundle["feature_cols"]
    if not isinstance(feature_cols, list) or not feature_cols or any(not isinstance(c, str) or not c for c in feature_cols):
        raise RuntimeError("Production model bundle feature_cols is invalid")
    weights = bundle["weights"]
    models = bundle["models"]
    if not isinstance(weights, dict) or not isinstance(models, dict) or not weights:
        raise RuntimeError("Production model bundle weights/models are invalid")
    if set(weights) != set(models):
        raise RuntimeError("Production model bundle weights/models mismatch")
    numeric_weights = np.asarray(list(weights.values()), dtype=float)
    if not np.all(np.isfinite(numeric_weights)) or np.any(numeric_weights < 0):
        raise RuntimeError("Production model bundle contains invalid ensemble weights")
    if not np.isclose(float(numeric_weights.sum()), 1.0, atol=1e-8):
        raise RuntimeError("Production model bundle ensemble weights are not normalized")
    if bundle["schema_version"] >= 2 and "score_model" not in bundle:
        raise RuntimeError("Production model bundle schema 2 requires score_model")
    score_method = str(bundle.get("score_method", "primary"))
    if score_method not in {"primary", "recency", "time_decay", "dixon_coles"}:
        raise RuntimeError(f"Unsupported production score method: {score_method}")
    if bundle["schema_version"] >= 2:
        score_model = bundle.get("score_model")
        if not isinstance(score_model, dict):
            raise RuntimeError("Production model bundle score_model must be an object")
        method = str(score_model.get("method", ""))
        expected_prefix = {
            "primary": "pit_smoothed_",
            "recency": "pit_recency_weighted_",
            "time_decay": "pit_time_decay_weighted_",
            "dixon_coles": "dixon_coles_",
        }[score_method]
        if not method.startswith(expected_prefix):
            raise RuntimeError(
                f"Production score method/model mismatch: score_method={score_method!r}, model_method={method!r}"
            )
        if score_method == "time_decay":
            try:
                half_life_days = float(score_model.get("half_life_days"))
            except (TypeError, ValueError):
                raise RuntimeError("Production time-decay score model has invalid half_life_days")
            if not np.isfinite(half_life_days) or half_life_days <= 0:
                raise RuntimeError("Production time-decay score model has non-positive half_life_days")
        if score_method != "primary":
            gate = bundle.get("score_locked_verification")
            if not isinstance(gate, dict) or gate.get("status") != "PASS" or str(gate.get("selected_method")) != score_method:
                raise RuntimeError("Non-primary score method requires matching PASS locked-OOS verification")
    temperature = float(bundle["temperature"])
    if not np.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("Production model bundle temperature is invalid")
    if not isinstance(bundle.get("fit_rows"), (int, np.integer)) or int(bundle["fit_rows"]) <= 0:
        raise RuntimeError("Production model bundle fit_rows is invalid")
    return bundle


def predict_bundle(bundle: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    feature_cols = bundle["feature_cols"]
    missing = [c for c in feature_cols if c not in X.columns]
    if missing:
        raise RuntimeError(f"Prediction input missing model features: {missing[:10]}")
    probs = np.zeros((len(X), 3), dtype=float)
    for name, model in bundle["models"].items():
        probs += float(bundle["weights"][name]) * model.predict_proba(X[feature_cols])
    probs = np.clip(probs, 1e-9, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    return _temperature_transform(probs, float(bundle["temperature"]))
