from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.prediction.secondary_outputs import _score_lambdas


def dixon_coles_tau(
    home_goals: int,
    away_goals: int,
    home_lambda: float,
    away_lambda: float,
    rho: float,
) -> float:
    """Dixon-Coles low-score dependence correction."""
    h, a = int(home_goals), int(away_goals)
    lh, la, r = float(home_lambda), float(away_lambda), float(rho)
    if h == 0 and a == 0:
        return 1.0 - lh * la * r
    if h == 0 and a == 1:
        return 1.0 + lh * r
    if h == 1 and a == 0:
        return 1.0 + la * r
    if h == 1 and a == 1:
        return 1.0 - r
    return 1.0


def dixon_coles_distribution(
    home_lambda: float,
    away_lambda: float,
    rho: float,
    *,
    max_goals: int = 12,
) -> list[tuple[int, int, float]]:
    """Normalized independent-Poisson score distribution with DC correction."""
    if max_goals < 1:
        raise ValueError("max_goals must be at least 1")
    lh = float(np.clip(home_lambda, 1e-6, 8.0))
    la = float(np.clip(away_lambda, 1e-6, 8.0))
    r = float(rho)

    cells: list[tuple[int, int, float]] = []
    for h in range(max_goals + 1):
        ph = math.exp(-lh) * lh**h / math.factorial(h)
        for a in range(max_goals + 1):
            pa = math.exp(-la) * la**a / math.factorial(a)
            tau = dixon_coles_tau(h, a, lh, la, r)
            if not math.isfinite(tau) or tau <= 0.0:
                raise ValueError("Invalid Dixon-Coles correction factor")
            cells.append((h, a, ph * pa * tau))

    normalizer = sum(p for _, _, p in cells)
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("Invalid Dixon-Coles normalizer")
    return sorted(
        [(h, a, p / normalizer) for h, a, p in cells],
        key=lambda x: (-float(x[2]), int(x[0]), int(x[1])),
    )


def _safe_rho_bounds(
    lambdas: list[tuple[float, float]],
    requested: float = 0.30,
) -> tuple[float, float]:
    """Bound rho so all low-score tau terms stay positive with margin."""
    upper = min(float(requested), 0.30)
    for lh, la in lambdas:
        upper = min(upper, 0.95 / max(lh, 1e-6), 0.95 / max(la, 1e-6))
    upper = max(0.02, min(upper, 0.30))
    return -upper, upper


def fit_dixon_coles_model(
    history: pd.DataFrame,
    *,
    shrinkage: float = 20.0,
    half_life_rows: float = 800.0,
    rho_l2: float = 0.02,
) -> dict[str, Any]:
    """Fit only the DC rho correction over the existing PIT score-rate model.

    The base team rates are exactly the production score-rate estimator; rho is
    learned from earlier PIT-verified matches with deterministic recency weights.
    This is a challenger and never changes the Champion on its own.
    """
    required = {
        "kickoff_utc",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "pit_verified",
    }
    missing = sorted(required - set(history.columns))
    if missing:
        raise ValueError(f"Dixon-Coles training data missing columns: {missing}")

    base = __import__(
        "src.prediction.secondary_outputs",
        fromlist=["fit_score_rate_model"],
    ).fit_score_rate_model(history, shrinkage=shrinkage)

    d = history.loc[history["pit_verified"] == True].copy()
    d["kickoff_utc"] = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    d["home_goals"] = pd.to_numeric(d["home_goals"], errors="coerce")
    d["away_goals"] = pd.to_numeric(d["away_goals"], errors="coerce")
    d = d.dropna(
        subset=["kickoff_utc", "home_goals", "away_goals", "home_team", "away_team"]
    ).sort_values("kickoff_utc", kind="mergesort").reset_index(drop=True)
    if d.empty:
        raise ValueError("No PIT-verified rows available for Dixon-Coles rho fit")

    lambdas: list[tuple[float, float]] = []
    outcomes: list[tuple[int, int]] = []
    weights: list[float] = []
    n = len(d)
    half = max(float(half_life_rows), 1.0)

    for pos, row in enumerate(d.itertuples(index=False)):
        comp = getattr(row, "competition", None)
        lh, la = _score_lambdas(base, row.home_team, row.away_team, comp)
        lambdas.append((lh, la))
        outcomes.append((int(row.home_goals), int(row.away_goals)))
        weights.append(math.exp((pos - (n - 1)) / half))

    weights_arr = np.asarray(weights, dtype=float)
    weights_arr /= max(float(weights_arr.mean()), 1e-12)
    lo, hi = _safe_rho_bounds(lambdas)

    def objective(rho: float) -> float:
        total = 0.0
        for (lh, la), (h, a), w in zip(lambdas, outcomes, weights_arr):
            ph = math.exp(-lh) * lh**h / math.factorial(h)
            pa = math.exp(-la) * la**a / math.factorial(a)
            tau = dixon_coles_tau(h, a, lh, la, rho)
            if tau <= 0.0 or not math.isfinite(tau):
                return 1e12
            total -= float(w) * math.log(max(ph * pa * tau, 1e-300))
        total /= max(float(weights_arr.sum()), 1.0)
        total += float(rho_l2) * float(rho * rho)
        return float(total)

    result = minimize_scalar(
        objective,
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 0.002},
    )
    rho = float(result.x) if result.success and math.isfinite(float(result.x)) else 0.0
    baseline_loss = objective(0.0)
    fitted_loss = objective(rho)
    used = bool(result.success and fitted_loss + 1e-8 < baseline_loss)

    return {
        "schema_version": 1,
        "method": "dixon_coles_low_score_correction_over_pit_score_rates",
        "base_model": base,
        "rho": float(rho if used else 0.0),
        "rho_fit_used": used,
        "rho_bounds": [float(lo), float(hi)],
        "half_life_rows": float(half),
        "rho_l2": float(rho_l2),
        "training_rows": int(len(d)),
        "baseline_objective": float(baseline_loss),
        "fitted_objective": float(fitted_loss),
    }


def predict_dixon_coles_distribution(
    model: dict[str, Any],
    home_team: str,
    away_team: str,
    competition: str | None = None,
    *,
    max_goals: int = 12,
) -> list[tuple[int, int, float]]:
    base = model.get("base_model")
    if not isinstance(base, dict):
        raise ValueError("Dixon-Coles model missing base_model")
    lh, la = _score_lambdas(base, home_team, away_team, competition)
    return dixon_coles_distribution(
        lh,
        la,
        float(model.get("rho", 0.0)),
        max_goals=max_goals,
    )
