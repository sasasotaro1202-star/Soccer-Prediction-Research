"""PIT-safe adaptive model routing challenger.

Routing weights are learned only from historical OOS evidence. The router is
never an adoption gate and must not be allowed to override a locked production
decision without separate evidence.
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

@dataclass(frozen=True)
class RouterConfig:
    min_context_rows: int = 120
    shrinkage_rows: int = 360
    max_weight_delta: float = 0.20
    min_model_weight: float = 0.05
    max_model_weight: float = 0.70
    decay_days: float = 180.0

def _normalize(weights: Mapping[str, float]) -> dict[str, float]:
    values = {str(k): float(v) for k, v in weights.items()}
    if not values or any(not np.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError("Weights must be finite and non-negative")
    total = sum(values.values())
    if total <= 0:
        raise ValueError("Weight sum must be positive")
    return {k: v / total for k, v in values.items()}

def _logloss(y: np.ndarray, probs: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(y) != len(p):
        raise ValueError("Expected N x 3 probabilities and N outcomes")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Invalid probability distribution")
    if not np.isin(y, (0, 1, 2)).all():
        raise ValueError("Outcomes must be encoded 0=H, 1=D, 2=A")
    loss = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))
    if sample_weight is None:
        return float(loss.mean())
    w = np.asarray(sample_weight, dtype=float)
    if len(w) != len(loss) or not np.isfinite(w).all() or (w < 0).any() or float(w.sum()) <= 0:
        raise ValueError("Invalid sample weights")
    return float(np.average(loss, weights=w))

class AdaptiveModelRouter:
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
        base_weights = base_weights or {m: 1.0 / len(self.models) for m in self.models}
        if set(base_weights) != set(self.models):
            raise ValueError("base_weights must cover every configured model")
        self.base_weights = _normalize(base_weights)
        self._global_weights = dict(self.base_weights)
        self._context_weights: dict[tuple[str, ...], dict[str, float]] = {}
        self._context_counts: dict[tuple[str, ...], int] = {}
        self.context_columns: tuple[str, ...] = ()

    def _bounded(self, target: Mapping[str, float], anchor: Mapping[str, float]) -> dict[str, float]:
        c = self.config
        values = {}
        for model in self.models:
            a = float(anchor[model])
            t = float(target.get(model, a))
            t = min(max(t, a - c.max_weight_delta), a + c.max_weight_delta)
            values[model] = min(max(t, c.min_model_weight), c.max_model_weight)
        return _normalize(values)

    def fit(
        self,
        oos: pd.DataFrame,
        *,
        context_columns: Sequence[str] | None = None,
        as_of: str | pd.Timestamp | None = None,
    ) -> "AdaptiveModelRouter":
        required = {"prediction_time_utc", "y", "model", "p_home", "p_draw", "p_away"}
        if not required.issubset(oos.columns):
            raise ValueError(f"OOS routing evidence missing columns: {sorted(required - set(oos.columns))}")
        d = oos.copy()
        d["prediction_time_utc"] = pd.to_datetime(d["prediction_time_utc"], utc=True, errors="coerce")
        if d["prediction_time_utc"].isna().any():
            raise ValueError("Routing evidence contains invalid prediction timestamps")
        cutoff = d["prediction_time_utc"].max() if as_of is None else pd.Timestamp(as_of)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        if "pit_verified" in d.columns:
            d["pit_verified"] = d["pit_verified"].astype("boolean")
            d = d[(d["prediction_time_utc"] < cutoff) & (d["pit_verified"] == True)]
        else:
            if "outcome_available_at_utc" not in d.columns:
                raise ValueError("Routing evidence needs pit_verified or outcome_available_at_utc")
            d["outcome_available_at_utc"] = pd.to_datetime(d["outcome_available_at_utc"], utc=True, errors="coerce")
            d = d[(d["prediction_time_utc"] < cutoff) & (d["outcome_available_at_utc"] < cutoff)]
        d = d[d["model"].astype(str).isin(self.models)].copy()
        d["y"] = pd.to_numeric(d["y"], errors="coerce")
        for c in ("p_home", "p_draw", "p_away"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["y", "p_home", "p_draw", "p_away"])
        if d.empty or not d["y"].isin((0, 1, 2)).all():
            raise ValueError("No valid PIT-safe OOS routing evidence")
        probs = d[["p_home", "p_draw", "p_away"]].to_numpy(float)
        if (probs < 0).any() or (probs > 1).any() or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Invalid OOS routing probabilities")
        c = self.config
        if c.decay_days <= 0 or not np.isfinite(c.decay_days):
            raise ValueError("decay_days must be positive and finite")
        age_days = (cutoff - d["prediction_time_utc"]).dt.total_seconds() / 86400.0
        d["_recency_weight"] = np.exp(-np.maximum(age_days, 0.0) / c.decay_days).clip(lower=1e-6)
        self.context_columns = tuple(x for x in (context_columns or DEFAULT_CONTEXT_COLUMNS) if x in d.columns)
        self._context_weights = {}
        self._context_counts = {}
        global_scores = {}
        for model in self.models:
            sample = d[d["model"].astype(str) == model]
            global_scores[model] = _logloss(
                sample["y"].to_numpy(int),
                sample[["p_home", "p_draw", "p_away"]].to_numpy(float),
                sample["_recency_weight"].to_numpy(float),
            ) if not sample.empty else 1.0
        self._global_weights = self._bounded(
            _normalize({m: 1.0 / max(v, 1e-6) for m, v in global_scores.items()}),
            self.base_weights,
        )
        for key, group in d.groupby(list(self.context_columns), dropna=False, sort=False) if self.context_columns else []:
            key = key if isinstance(key, tuple) else (key,)
            key = tuple(str(x) for x in key)
            n = int(group["prediction_time_utc"].nunique())
            self._context_counts[key] = n
            if n < c.min_context_rows:
                continue
            scores = {}
            for model in self.models:
                sample = group[group["model"].astype(str) == model]
                scores[model] = _logloss(
                    sample["y"].to_numpy(int),
                    sample[["p_home", "p_draw", "p_away"]].to_numpy(float),
                    sample["_recency_weight"].to_numpy(float),
                ) if not sample.empty else global_scores[model]
            inverse = _normalize({m: 1.0 / max(v, 1e-6) for m, v in scores.items()})
            alpha = min(1.0, n / max(c.shrinkage_rows, 1))
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
        missing = [m for m in self.models if m not in candidate_probabilities]
        if missing:
            raise ValueError(f"Candidate probabilities missing models: {missing}")
        arrays = {m: np.asarray(candidate_probabilities[m], dtype=float) for m in self.models}
        shape = next(iter(arrays.values())).shape
        if len(shape) != 2 or shape[1] != 3 or any(a.shape != shape for a in arrays.values()):
            raise ValueError("Candidate probabilities must all be N x 3")
        if any(not np.isfinite(a).all() or (a < 0).any() or (a > 1).any() for a in arrays.values()):
            raise ValueError("Candidate probabilities contain invalid values")
        if any(not np.allclose(a.sum(axis=1), 1.0, atol=1e-6) for a in arrays.values()):
            raise ValueError("Candidate probabilities must sum to one")
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
