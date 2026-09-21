"""PIT-safe adaptive weighting for the soccer candidate model set.

This module is a challenger only. It can learn context-specific mixture weights
from historical OOS predictions, but it never decides production adoption.
Adoption remains a separate OOS/locked-holdout gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_MODELS = ("logistic", "logistic_select", "extra_trees", "random_forest", "hist_gb")
DEFAULT_CONTEXT_COLUMNS = (
    "competition",
    "strength_gap_bin",
    "scoring_environment_bin",
    "rest_bin",
    "starter_status",
    "odds_missing",
)
REQUIRED_OOS_COLUMNS = {
    "prediction_time_utc",
    "outcome_available_at_utc",
    "y",
    "model",
    "p_home",
    "p_draw",
    "p_away",
}


def _log_loss(y: np.ndarray, probs: np.ndarray, weight: np.ndarray | None = None) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(y) != len(p):
        raise ValueError("Expected N x 3 probabilities and N outcomes")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Probabilities must be finite, non-negative and sum to one")
    if not np.isin(y, (0, 1, 2)).all():
        raise ValueError("Outcomes must be encoded as 0=H, 1=D, 2=A")
    loss = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))
    if weight is None:
        return float(loss.mean())
    w = np.asarray(weight, dtype=float)
    if len(w) != len(loss) or not np.isfinite(w).all() or (w < 0).any() or float(w.sum()) <= 0:
        raise ValueError("Weights must be finite, non-negative and non-zero")
    return float(np.average(loss, weights=w))


@dataclass(frozen=True)
class RouterConfig:
    min_context_rows: int = 120
    shrinkage_rows: int = 360
    max_weight_delta: float = 0.20
    min_model_weight: float = 0.05
    max_model_weight: float = 0.70
    decay_days: float = 180.0


class AdaptiveModelRouter:
    """Learn deterministic model-mixture weights using only prior OOS evidence."""

    def __init__(
        self,
        models: Sequence[str] = DEFAULT_MODELS,
        base_weights: Mapping[str, float] | None = None,
        config: RouterConfig | None = None,
    ) -> None:
        self.models = tuple(dict.fromkeys(str(x) for x in models))
        if len(self.models) < 2:
            raise ValueError("Adaptive routing requires at least two models")
        self.config = config or RouterConfig()
        if base_weights is None:
            base_weights = {m: 1.0 / len(self.models) for m in self.models}
        if set(base_weights) != set(self.models):
            raise ValueError("base_weights must cover every configured model")
        self.base_weights = self._normalize(base_weights)
        self._global_weights = dict(self.base_weights)
        self._context_weights: dict[tuple[str, ...], dict[str, float]] = {}
        self._context_counts: dict[tuple[str, ...], int] = {}
        self.context_columns: tuple[str, ...] = ()

    @staticmethod
    def _normalize(weights: Mapping[str, float]) -> dict[str, float]:
        values = {str(k): float(v) for k, v in weights.items()}
        if not values or any(not np.isfinite(v) or v < 0 for v in values.values()):
            raise ValueError("Weights must be finite and non-negative")
        total = sum(values.values())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Weight sum must be positive")
        return {k: v / total for k, v in values.items()}

    def _bounded(self, target: Mapping[str, float], anchor: Mapping[str, float]) -> dict[str, float]:
        c = self.config
        values: dict[str, float] = {}
        for model in self.models:
            anchor_value = float(anchor.get(model, 0.0))
            target_value = float(target.get(model, anchor_value))
            target_value = min(max(target_value, anchor_value - c.max_weight_delta), anchor_value + c.max_weight_delta)
            values[model] = min(max(target_value, c.min_model_weight), c.max_model_weight)
        return self._normalize(values)

    def fit(
        self,
        oos: pd.DataFrame,
        *,
        context_columns: Sequence[str] | None = None,
        as_of: str | pd.Timestamp | None = None,
    ) -> "AdaptiveModelRouter":
        missing = sorted(REQUIRED_OOS_COLUMNS - set(oos.columns))
        if missing:
            raise ValueError(f"OOS routing evidence missing columns: {missing}")
        d = oos.copy()
        d["prediction_time_utc"] = pd.to_datetime(d["prediction_time_utc"], utc=True, errors="coerce")
        d["outcome_available_at_utc"] = pd.to_datetime(d["outcome_available_at_utc"], utc=True, errors="coerce")
        if d[["prediction_time_utc", "outcome_available_at_utc"]].isna().any().any():
            raise ValueError("OOS routing evidence contains invalid timestamps")
        if as_of is None:
            cutoff = d["prediction_time_utc"].max()
        else:
            cutoff = pd.Timestamp(as_of)
            cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        if pd.isna(cutoff):
            raise ValueError("No valid routing cutoff")

        d = d[
            (d["prediction_time_utc"] < cutoff)
            & (d["outcome_available_at_utc"] < cutoff)
            & (d["outcome_available_at_utc"] >= d["prediction_time_utc"])
            & d["model"].astype(str).isin(self.models)
        ].copy()
        if d.empty:
            raise ValueError("No PIT-safe historical OOS routing evidence")
        d["y"] = pd.to_numeric(d["y"], errors="coerce")
        for col in ("p_home", "p_draw", "p_away"):
            d[col] = pd.to_numeric(d[col], errors="coerce")
        d = d.dropna(subset=["y", "p_home", "p_draw", "p_away"])
        if d.empty or not d["y"].isin((0, 1, 2)).all():
            raise ValueError("Routing outcomes must be valid 1X2 classes")
        probs = d[["p_home", "p_draw", "p_away"]].to_numpy(float)
        if (probs < 0).any() or (probs > 1).any() or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Routing probabilities must be valid distributions")
        if self.config.decay_days <= 0 or not np.isfinite(self.config.decay_days):
            raise ValueError("decay_days must be positive and finite")
        age_days = (cutoff - d["prediction_time_utc"]).dt.total_seconds() / 86400.0
        d["_recency_weight"] = np.exp(-np.maximum(age_days, 0.0) / self.config.decay_days).clip(lower=1e-6)

        self.context_columns = tuple(c for c in (context_columns or DEFAULT_CONTEXT_COLUMNS) if c in d.columns)
        self._context_weights = {}
        self._context_counts = {}

        global_scores = {}
        for model in self.models:
            sample = d[d["model"].astype(str) == model]
            global_scores[model] = (
                _log_loss(
                    sample["y"].to_numpy(int),
                    sample[["p_home", "p_draw", "p_away"]].to_numpy(float),
                    sample["_recency_weight"].to_numpy(float),
                )
                if not sample.empty else 1.0
            )
        self._global_weights = self._bounded(
            self._normalize({m: 1.0 / max(s, 1e-6) for m, s in global_scores.items()}),
            self.base_weights,
        )

        for key, group in d.groupby(list(self.context_columns), dropna=False, sort=False) if self.context_columns else []:
            if not isinstance(key, tuple):
                key = (key,)
            key = tuple(str(x) for x in key)
            n = int(group["prediction_time_utc"].nunique())
            self._context_counts[key] = n
            if n < self.config.min_context_rows:
                continue
            scores = {}
            for model in self.models:
                sample = group[group["model"].astype(str) == model]
                scores[model] = (
                    _log_loss(
                        sample["y"].to_numpy(int),
                        sample[["p_home", "p_draw", "p_away"]].to_numpy(float),
                        sample["_recency_weight"].to_numpy(float),
                    )
                    if not sample.empty else global_scores[model]
                )
            inverse = self._normalize({m: 1.0 / max(s, 1e-6) for m, s in scores.items()})
            alpha = min(1.0, n / max(self.config.shrinkage_rows, 1))
            blended = {m: alpha * inverse[m] + (1.0 - alpha) * self._global_weights[m] for m in self.models}
            self._context_weights[key] = self._bounded(blended, self._global_weights)
        return self

    def route(
        self,
        context: Mapping[str, object],
        candidate_probabilities: Mapping[str, Sequence[float] | np.ndarray],
        *,
        context_columns: Sequence[str] | None = None,
    ) -> dict[str, object]:
        if not self._global_weights:
            raise RuntimeError("Router must be fitted before routing")
        missing = [m for m in self.models if m not in candidate_probabilities]
        if missing:
            raise ValueError(f"Candidate probabilities missing models: {missing}")
        arrays = {m: np.asarray(candidate_probabilities[m], dtype=float) for m in self.models}
        shape = next(iter(arrays.values())).shape
        if len(shape) != 2 or shape[1] != 3 or any(a.shape != shape for a in arrays.values()):
            raise ValueError("Candidate probabilities must all be shaped N x 3")
        if any(not np.isfinite(a).all() or (a < 0).any() or (a > 1).any() for a in arrays.values()):
            raise ValueError("Candidate probabilities contain invalid values")
        if any(not np.allclose(a.sum(axis=1), 1.0, atol=1e-6) for a in arrays.values()):
            raise ValueError("Candidate probability rows must sum to one")
        columns = tuple(context_columns or self.context_columns)
        key = tuple(str(context.get(c, "__MISSING__")) for c in columns)
        weights = self._context_weights.get(key, self._global_weights)
        matrix = np.stack([arrays[m] for m in self.models], axis=0)
        blended = np.average(matrix, axis=0, weights=[weights[m] for m in self.models])
        return {
            "probabilities": blended,
            "weights": dict(weights),
            "context_key": key,
            "context_sample_count": int(self._context_counts.get(key, 0)),
            "routing_source": "CONTEXT_OOS" if key in self._context_weights else "GLOBAL_OOS",
            "models": list(self.models),
        }
