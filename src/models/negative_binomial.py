from __future__ import annotations

"""PIT-safe Negative-Binomial score challenger.

The model keeps the existing PIT-smoothed team scoring-rate structure and changes
only the goal-count distribution: each venue-side goal count follows a
Negative-Binomial distribution with dispersion estimated from PIT-verified
training outcomes. It is a challenger only; adoption is controlled elsewhere.
"""

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import gammaln

from src.prediction.secondary_outputs import _score_lambdas, fit_score_rate_model


def _estimate_dispersion(values: pd.Series, *, prior: float = 50.0, prior_strength: float = 120.0) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 2:
        return float(prior)
    mean = float(np.mean(x))
    var = float(np.var(x, ddof=1))
    mean = max(mean, 1e-6)
    if not np.isfinite(var) or var <= mean + 1e-9:
        raw = float(prior)
    else:
        raw = float((mean * mean) / max(var - mean, 1e-9))
    raw = float(np.clip(raw, 2.0, 1000.0))
    alpha = float(len(x) / (len(x) + max(float(prior_strength), 1.0)))
    return float(np.exp(alpha * np.log(raw) + (1.0 - alpha) * np.log(float(prior))))


def _nb_pmf(k: int, mean: float, dispersion: float) -> float:
    if k < 0:
        return 0.0
    mu = max(float(mean), 1e-9)
    r = max(float(dispersion), 1e-6)
    log_p = (
        gammaln(k + r)
        - gammaln(r)
        - gammaln(k + 1.0)
        + r * math.log(r / (r + mu))
        + k * math.log(mu / (r + mu))
    )
    return float(math.exp(log_p))


def negative_binomial_distribution(
    home_lambda: float,
    away_lambda: float,
    home_dispersion: float,
    away_dispersion: float,
    *,
    max_goals: int = 12,
) -> list[tuple[int, int, float]]:
    if max_goals < 1:
        raise ValueError("max_goals must be at least 1")
    cells = []
    for home_goals in range(max_goals + 1):
        ph = _nb_pmf(home_goals, home_lambda, home_dispersion)
        for away_goals in range(max_goals + 1):
            pa = _nb_pmf(away_goals, away_lambda, away_dispersion)
            cells.append((home_goals, away_goals, ph * pa))
    z = float(sum(p for _, _, p in cells))
    if not np.isfinite(z) or z <= 0:
        raise RuntimeError("Negative-Binomial score distribution normalization failed")
    out = [(h, a, float(p / z)) for h, a, p in cells]
    out.sort(key=lambda x: (-x[2], x[0], x[1]))
    if not np.isfinite(np.asarray([p for _, _, p in out], dtype=float)).all():
        raise RuntimeError("Negative-Binomial score distribution contains non-finite probabilities")
    return out


def fit_negative_binomial_score_model(
    history: pd.DataFrame,
    *,
    shrinkage: float = 20.0,
    dispersion_prior: float = 50.0,
    dispersion_prior_strength: float = 120.0,
) -> dict[str, Any]:
    required = {"home_team", "away_team", "home_goals", "away_goals", "pit_verified"}
    missing = sorted(required - set(history.columns))
    if missing:
        raise ValueError(f"Negative-Binomial training data missing columns: {missing}")
    d = history.loc[history["pit_verified"] == True].copy()
    d["home_goals"] = pd.to_numeric(d["home_goals"], errors="coerce")
    d["away_goals"] = pd.to_numeric(d["away_goals"], errors="coerce")
    d = d.dropna(subset=["home_goals", "away_goals", "home_team", "away_team"])
    if d.empty:
        raise ValueError("No PIT-verified score rows available")
    base = fit_score_rate_model(d, shrinkage=shrinkage)
    home_dispersion = _estimate_dispersion(
        d["home_goals"],
        prior=dispersion_prior,
        prior_strength=dispersion_prior_strength,
    )
    away_dispersion = _estimate_dispersion(
        d["away_goals"],
        prior=dispersion_prior,
        prior_strength=dispersion_prior_strength,
    )
    return {
        "schema_version": 1,
        "method": "negative_binomial_pit_overdispersed_team_rates",
        "base_model": base,
        "home_dispersion": home_dispersion,
        "away_dispersion": away_dispersion,
        "dispersion_prior": float(dispersion_prior),
        "dispersion_prior_strength": float(dispersion_prior_strength),
        "training_rows": int(len(d)),
    }


def predict_negative_binomial_distribution(
    model: dict[str, Any],
    home_team: str,
    away_team: str,
    competition: str | None = None,
    *,
    max_goals: int = 12,
) -> list[tuple[int, int, float]]:
    base = model.get("base_model")
    if not isinstance(base, dict):
        raise ValueError("Negative-Binomial model missing base_model")
    home_lambda, away_lambda = _score_lambdas(base, home_team, away_team, competition)
    return negative_binomial_distribution(
        home_lambda,
        away_lambda,
        float(model.get("home_dispersion", 50.0)),
        float(model.get("away_dispersion", 50.0)),
        max_goals=max_goals,
    )
