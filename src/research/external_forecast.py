from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

PROBABILITY_COLUMNS = ("p_home", "p_draw", "p_away")
REQUIRED_COLUMNS = (
    "match_id",
    "source",
    "prediction_time_utc",
    "source_available_at_utc",
    *PROBABILITY_COLUMNS,
    "provenance_url",
)
DEFAULT_WEIGHT_GRID = tuple(np.linspace(0.0, 1.0, 21))


class ExternalForecastValidationError(ValueError):
    """Raised when an external forecast cannot be used under the research contract."""


@dataclass(frozen=True)
class ExternalForecastDiagnostics:
    rows: int
    source_count: int
    pit_verified_rows: int
    mean_disagreement_l1: float
    top_class_disagreement_rate: float


def _as_probability_matrix(values: Iterable[Iterable[float]], *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ExternalForecastValidationError(f"{name} must have shape (n, 3)")
    if len(matrix) == 0:
        raise ExternalForecastValidationError(f"{name} must not be empty")
    if not np.isfinite(matrix).all():
        raise ExternalForecastValidationError(f"{name} contains NaN/Inf")
    if np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ExternalForecastValidationError(f"{name} contains out-of-range probabilities")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6):
        raise ExternalForecastValidationError(f"{name} rows must sum to 1")
    return matrix


def validate_external_forecasts(
    frame: pd.DataFrame,
    *,
    require_prediction_time: bool = True,
) -> pd.DataFrame:
    """Validate an external 1X2 forecast table and fail closed on PIT uncertainty.

    External sources are treated as research evidence only. A forecast row is
    usable only when source_available_at_utc <= prediction_time_utc, with both
    timestamps known. Provenance is mandatory so the forecast can be replayed
    and audited later.
    """
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ExternalForecastValidationError(f"missing required columns: {missing}")

    df = frame.copy()
    for col in ("prediction_time_utc", "source_available_at_utc"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    if require_prediction_time and df["prediction_time_utc"].isna().any():
        raise ExternalForecastValidationError("unknown prediction_time_utc is FAIL-CLOSED")
    if df["source_available_at_utc"].isna().any():
        raise ExternalForecastValidationError("unknown source_available_at_utc is FAIL-CLOSED")
    if (df["source_available_at_utc"] > df["prediction_time_utc"]).any():
        raise ExternalForecastValidationError("external forecast violates PIT: source available after prediction")

    df["match_id"] = df["match_id"].astype(str).str.strip()
    df["source"] = df["source"].astype(str).str.strip()
    df["provenance_url"] = df["provenance_url"].astype(str).str.strip()
    if (df["match_id"] == "").any() or (df["source"] == "").any() or (df["provenance_url"] == "").any():
        raise ExternalForecastValidationError("match_id/source/provenance_url must be non-empty")

    probs = _as_probability_matrix(df[list(PROBABILITY_COLUMNS)].to_numpy(), name="external probabilities")
    df.loc[:, PROBABILITY_COLUMNS] = probs

    dup = df.duplicated(
        subset=["match_id", "source", "prediction_time_utc"],
        keep=False,
    )
    if dup.any():
        raise ExternalForecastValidationError("duplicate external forecast identity detected")

    df["pit_status"] = "PIT_VERIFIED"
    df["top_class"] = np.asarray(["H", "D", "A"])[np.argmax(probs, axis=1)]
    df["confidence"] = probs.max(axis=1)
    return df


def disagreement_features(base_probs: Iterable[Iterable[float]], external_probs: Iterable[Iterable[float]]) -> pd.DataFrame:
    """Return deterministic disagreement features for a base/external pair."""
    base = _as_probability_matrix(base_probs, name="base_probs")
    ext = _as_probability_matrix(external_probs, name="external_probs")
    if len(base) != len(ext):
        raise ExternalForecastValidationError("base_probs and external_probs row counts differ")

    l1 = np.abs(base - ext).sum(axis=1)
    base_top = np.argmax(base, axis=1)
    ext_top = np.argmax(ext, axis=1)
    return pd.DataFrame(
        {
            "external_disagreement_l1": l1,
            "external_top_class_disagreement": base_top != ext_top,
            "external_base_confidence": base.max(axis=1),
            "external_source_confidence": ext.max(axis=1),
            "external_confidence_delta": ext.max(axis=1) - base.max(axis=1),
        }
    )


def linear_pool(base_probs: Iterable[Iterable[float]], external_probs: Iterable[Iterable[float]], weight: float) -> np.ndarray:
    """Fuse two probability forecasts with a convex linear pool."""
    base = _as_probability_matrix(base_probs, name="base_probs")
    ext = _as_probability_matrix(external_probs, name="external_probs")
    if len(base) != len(ext):
        raise ExternalForecastValidationError("base_probs and external_probs row counts differ")
    weight = float(weight)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ExternalForecastValidationError("weight must be in [0, 1]")

    fused = (1.0 - weight) * base + weight * ext
    fused /= fused.sum(axis=1, keepdims=True)
    if not np.isfinite(fused).all() or not np.allclose(fused.sum(axis=1), 1.0, atol=1e-6):
        raise ExternalForecastValidationError("linear pool produced invalid probabilities")
    return fused


def _multiclass_logloss(y_true: Iterable[int], probs: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    if len(y) != len(probs) or len(y) == 0:
        raise ExternalForecastValidationError("y_true/probabilities must be non-empty and aligned")
    if np.any((y < 0) | (y > 2)):
        raise ExternalForecastValidationError("y_true labels must be 0, 1, or 2")
    chosen = np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0)
    return float(-np.mean(np.log(chosen)))


def fit_linear_pool_weight(
    y_train: Iterable[int],
    base_probs_train: Iterable[Iterable[float]],
    external_probs_train: Iterable[Iterable[float]],
    *,
    weight_grid: Iterable[float] = DEFAULT_WEIGHT_GRID,
) -> dict[str, float]:
    """Fit an external-prior weight on a training slice only.

    The caller is responsible for chronological splitting. Never fit this weight
    using the evaluation, latest-unseen, or frozen/blind holdout period.
    """
    base = _as_probability_matrix(base_probs_train, name="base_probs_train")
    ext = _as_probability_matrix(external_probs_train, name="external_probs_train")
    if len(base) != len(ext):
        raise ExternalForecastValidationError("training probability row counts differ")
    y = np.asarray(y_train, dtype=int)
    if len(y) != len(base):
        raise ExternalForecastValidationError("training labels/probabilities are misaligned")

    candidates: list[tuple[float, float]] = []
    for raw_weight in weight_grid:
        weight = float(raw_weight)
        if not 0.0 <= weight <= 1.0:
            raise ExternalForecastValidationError("weight_grid contains a value outside [0, 1]")
        loss = _multiclass_logloss(y, linear_pool(base, ext, weight))
        candidates.append((loss, weight))

    best_loss, best_weight = min(candidates, key=lambda item: (item[0], abs(item[1])))
    return {
        "weight": float(best_weight),
        "train_logloss": float(best_loss),
        "train_rows": float(len(y)),
        "base_logloss": float(_multiclass_logloss(y, base)),
        "external_logloss": float(_multiclass_logloss(y, ext)),
    }


def build_diagnostics(
    base_probs: Iterable[Iterable[float]],
    external_probs: Iterable[Iterable[float]],
) -> ExternalForecastDiagnostics:
    features = disagreement_features(base_probs, external_probs)
    return ExternalForecastDiagnostics(
        rows=int(len(features)),
        source_count=1,
        pit_verified_rows=int(len(features)),
        mean_disagreement_l1=float(features["external_disagreement_l1"].mean()),
        top_class_disagreement_rate=float(features["external_top_class_disagreement"].mean()),
    )
