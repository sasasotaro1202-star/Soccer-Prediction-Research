from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.evaluation.metrics import classification_metrics
from src.research.external_forecast import (
    ExternalForecastValidationError,
    linear_pool,
    validate_external_forecasts,
)

RESULT_TO_LABEL = {"H": 0, "D": 1, "A": 2}
LABEL_TO_RESULT = np.array(["H", "D", "A"])
PROBABILITY_COLUMNS = ("p_home", "p_draw", "p_away")


def _probabilities(frame: pd.DataFrame, *, prefix: str = "") -> np.ndarray:
    cols = tuple(f"{prefix}{c}" for c in PROBABILITY_COLUMNS)
    missing = sorted(set(cols) - set(frame.columns))
    if missing:
        raise ExternalForecastValidationError(f"missing probability columns: {missing}")
    p = frame.loc[:, list(cols)].to_numpy(dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) == 0:
        raise ExternalForecastValidationError("probabilities must have shape (n, 3)")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ExternalForecastValidationError("probabilities contain invalid values")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ExternalForecastValidationError("probability rows must sum to 1")
    return p


def _labels(frame: pd.DataFrame, column: str = "actual_result") -> np.ndarray:
    if column not in frame.columns:
        raise ExternalForecastValidationError(f"missing outcome column: {column}")
    raw = frame[column].astype(str).str.strip().str.upper()
    unknown = sorted(set(raw) - set(RESULT_TO_LABEL))
    if unknown:
        raise ExternalForecastValidationError(f"unknown outcome labels: {unknown}")
    return raw.map(RESULT_TO_LABEL).to_numpy(dtype=int)


def evaluate_external_source(
    forecasts: pd.DataFrame,
    *,
    outcome_column: str = "actual_result",
) -> dict:
    """Validate PIT/provenance then calculate the standard 1X2 scorecard."""
    validated = validate_external_forecasts(forecasts)
    y = _labels(validated, outcome_column)
    p = _probabilities(validated)
    result = classification_metrics(y, p)
    result.update(
        {
            "source": str(validated["source"].iloc[0]) if len(validated) else "",
            "rows": int(len(validated)),
            "pit_verified_rows": int(validated["pit_status"].eq("PIT_VERIFIED").sum()),
            "prediction_time_min": validated["prediction_time_utc"].min().isoformat(),
            "prediction_time_max": validated["prediction_time_utc"].max().isoformat(),
        }
    )
    return result


def evaluate_internal_vs_external(
    forecasts: pd.DataFrame,
    *,
    base_prefix: str = "base_",
    outcome_column: str = "actual_result",
    blend_weight: float | None = None,
) -> dict:
    """Score external and internal forecasts on exactly the same PIT-verified rows.

    When blend_weight is supplied, it is applied after alignment and validation.
    The weight must have been selected on a separate chronological training slice.
    """
    validated = validate_external_forecasts(forecasts)
    y = _labels(validated, outcome_column)
    ext = _probabilities(validated)
    base = _probabilities(validated, prefix=base_prefix)

    external_metrics = classification_metrics(y, ext)
    internal_metrics = classification_metrics(y, base)

    out = {
        "rows": int(len(validated)),
        "source": str(validated["source"].iloc[0]) if len(validated) else "",
        "external": external_metrics,
        "internal": internal_metrics,
        "blend": None,
    }

    if blend_weight is not None:
        fused = linear_pool(base, ext, float(blend_weight))
        blend_metrics = classification_metrics(y, fused)
        out["blend"] = {
            "weight": float(blend_weight),
            **blend_metrics,
        }
    return out


def evaluate_multiple_sources(
    forecasts: pd.DataFrame,
    *,
    outcome_column: str = "actual_result",
    source_column: str = "source",
) -> dict[str, dict]:
    """Evaluate each source independently without pooling incompatible time rows."""
    if source_column not in forecasts.columns:
        raise ExternalForecastValidationError(f"missing source column: {source_column}")
    results: dict[str, dict] = {}
    for source, group in forecasts.groupby(source_column, sort=True):
        results[str(source)] = evaluate_external_source(group.copy(), outcome_column=outcome_column)
    return results
