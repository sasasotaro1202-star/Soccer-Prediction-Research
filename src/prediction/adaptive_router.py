"""PIT-safe, OOS-driven adaptive model routing for multiclass 1X2 probabilities.

The router is deliberately conservative:
- routing evidence must be available before the prediction timestamp;
- regime weights are shrunk toward global weights when samples are sparse;
- weights are bounded and normalized;
- no outcome from the current/future prediction horizon may influence routing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


REQUIRED_OOS_COLUMNS = {
    "prediction_time_utc",
    "outcome_available_at_utc",
    "y",
    "model",
    "p_home",
    "p_draw",
    "p_away",
}
DEFAULT_MODELS = ("ml_ensemble", "elo", "poisson", "market")
DEFAULT_CONTEXT_COLUMNS = (
    "competition",
    "strength_gap_bin",
    "scoring_environment_bin",
    "rest_bin",
    "starter_status",
    "odds_missing",
)


def _multiclass_log_loss(y: np.ndarray, probs: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(y) != len(p):
        raise ValueError("Expected N x 3 probability matrix and N outcomes")
    if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Probabilities must be finite and sum to one")
    if not np.isin(y, [0, 1, 2]).all():
        raise ValueError("1X2 outcomes must be encoded as 0=H, 1=D, 2=A")
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))))


@dataclass(frozen=True)
class RouterConfig:
    min_regime_samples: int = 120
    shrinkage_samples: int = 360
    max_weight_delta: float = 0.20
    min_model_weight: float = 0.05
    max_model_weight: float = 0.70
    decay_days: float = 180.0


class AdaptiveModelRouter:
    """Selects/weights candidate probability models using only historical OOS evidence."""

    def __init__(
        self,
        models: Sequence[str] = DEFAULT_MODELS,
        base_weights: Mapping[str, float] | None = None,
        config: RouterConfig | None = None,
    ) -> None:
        self.models = tuple(dict.fromkeys(str(x) for x in models))
        if len(self.models) < 2:
            raise ValueError("Adaptive routing requires at least two candidate models")
        self.config = config or RouterConfig()
        if base_weights is None:
            base_weights = {m: 1.0 / len(self.models) for m in self.models}
        self.base_weights = self._normalize(base_weights)
        self._global_weights = dict(self.base_weights)
        self._regime_weights: dict[tuple, dict[str, float]] = {}
        self._regime_counts: dict[tuple, int] = {}

    @staticmethod
    def _normalize(weights: Mapping[str, float]) -> dict[str, float]:
        vals = {str(k): float(v) for k, v in weights.items()}
        if not vals or not np.isfinite(list(vals.values())).all() or any(v < 0 for v in vals.values()):
            raise ValueError("Weights must be finite and non-negative")
        total = sum(vals.values())
        if total <= 0:
            raise ValueError("Weight sum must be positive")
        return {k: v / total for k, v in vals.items()}

    @staticmethod
    def _regime_key(row: Mapping[str, object], context_columns: Sequence[str]) -> tuple:
        return tuple(str(row.get(c, "__MISSING__")) for c in context_columns)

    def _bounded(self, target: Mapping[str, float], anchor: Mapping[str, float]) -> dict[str, float]:
        c = self.config
        out = {}
        for m in self.models:
            a = float(anchor.get(m, 0.0))
            t = float(target.get(m, a))
            t = min(max(t, a - c.max_weight_delta), a + c.max_weight_delta)
            out[m] = min(max(t, c.min_model_weight), c.max_model_weight)
        return self._normalize(out)

    def fit(
        self,
        oos: pd.DataFrame,
        *,
        context_columns: Sequence[str] = DEFAULT_CONTEXT_COLUMNS,
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
        # Hard PIT invariant: outcome evidence must predate the current routing cutoff.
        if as_of is not None:
            cutoff = pd.Timestamp(as_of)
            cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        else:
            cutoff = d["prediction_time_utc"].max()
        if cutoff is pd.NaT:
            raise ValueError("No valid routing cutoff")
        d = d[d["outcome_available_at_utc"] < cutoff].copy()
        d = d[d["model"].astype(str).isin(self.models)].copy()
        if d.empty:
            raise ValueError("No PIT-safe historical OOS routing evidence")
        d["y"] = pd.to_numeric(d["y"], errors="coerce")
        for col in ("p_home", "p_draw", "p_away"):
            d[col] = pd.to_numeric(d[col], errors="coerce")
        d = d.dropna(subset=["y", "p_home", "p_draw", "p_away"])
        if d.empty or not d["y"].isin([0, 1, 2]).all():
            raise ValueError("Routing outcomes must be encoded as 0=H, 1=D, 2=A")
        probs = d[["p_home", "p_draw", "p_away"]].to_numpy(float)
        if (probs < 0).any() or (probs > 1).any() or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("OOS routing probabilities must be valid 3-class distributions")

        global_scores: dict[str, float] = {}
        for model in self.models:
            x = d[d["model"].astype(str) == model]
            if x.empty:
                global_scores[model] = 1.0
            else:
                global_scores[model] = _multiclass_log_loss(x["y"].to_numpy(int), x[["p_home", "p_draw", "p_away"]].to_numpy(float))
        inv = {m: 1.0 / max(s, 1e-6) for m, s in global_scores.items()}
        self._global_weights = self._bounded(self._normalize(inv), self.base_weights)

        self._regime_weights = {}
        self._regime_counts = {}
        for key, group in d.groupby(list(context_columns), dropna=False, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            n = int(group["prediction_time_utc"].nunique())
            self._regime_counts[key] = n
            if n < self.config.min_regime_samples:
                continue
            scores = {}
            for model in self.models:
                x = group[group["model"].astype(str) == model]
                scores[model] = _log_loss(x["y"].to_numpy(), x["probability"].to_numpy()) if not x.empty else global_scores[model]
            inv_regime = self._normalize({m: 1.0 / max(s, 1e-6) for m, s in scores.items()})
            alpha = min(1.0, n / max(self.config.shrinkage_samples, 1))
            blended = {
                m: alpha * inv_regime[m] + (1.0 - alpha) * self._global_weights[m]
                for m in self.models
            }
            self._regime_weights[key] = self._bounded(blended, self._global_weights)
        return self

    def route(
        self,
        context: Mapping[str, object],
        candidate_probabilities: Mapping[str, Sequence[float]],
        *,
        context_columns: Sequence[str] = DEFAULT_CONTEXT_COLUMNS,
    ) -> dict[str, object]:
        if any(m not in candidate_probabilities for m in self.models):
            raise ValueError("Candidate probabilities are missing one or more configured models")
        arrays = {m: np.asarray(candidate_probabilities[m], dtype=float) for m in self.models}
        shape = next(iter(arrays.values())).shape
        if len(shape) != 2 or shape[1] != 3 or any(a.shape != shape for a in arrays.values()):
            raise ValueError("Candidate probabilities must be equally shaped N x 3 arrays")
        if any(not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("Candidate probabilities contain non-finite values")
        if any(((a < 0) | (a > 1)).any() for a in arrays.values()):
            raise ValueError("Candidate probabilities must be in [0, 1]")
        if any(not np.allclose(a.sum(axis=1), 1.0, atol=1e-6) for a in arrays.values()):
            raise ValueError("Each candidate probability row must sum to one")
        key = self._regime_key(context, context_columns)
        weights = self._regime_weights.get(key, self._global_weights)
        matrix = np.stack([arrays[m] for m in self.models], axis=0)
        blended = np.average(matrix, axis=0, weights=[weights[m] for m in self.models])
        return {
            "probabilities": blended,
            "weights": dict(weights),
            "regime_key": key,
            "regime_sample_count": int(self._regime_counts.get(key, 0)),
            "routing_source": "REGIME_OOS" if key in self._regime_weights else "GLOBAL_OOS",
            "models": list(self.models),
        }
