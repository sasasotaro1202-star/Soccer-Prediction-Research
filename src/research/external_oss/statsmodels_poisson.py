"""Research-only statsmodels Poisson score challenger.

This adapter intentionally implements only a bounded score-distribution
challenger. It does not modify the Production model bundle, feature policy,
target definitions, calibration, or frozen holdout.

PIT contract:
- only pit_verified rows are eligible;
- callers must provide the chronological training prefix already bounded by
  the prediction cutoff;
- no outcome-derived feature is created here.

The model is deliberately simple: separate Poisson GLMs for home and away
goals using categorical home team, away team, and competition terms.
Unknown categories at prediction time fail closed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except ImportError as exc:  # pragma: no cover - exercised by isolated envs
    raise ImportError(
        "statsmodels is a Research-only dependency. Install requirements-research.txt."
    ) from exc


_REQUIRED = {
    "home_team",
    "away_team",
    "competition",
    "home_goals",
    "away_goals",
    "pit_verified",
    "kickoff_utc",
    "source_available_at_utc",
}


@dataclass(frozen=True)
class StatsmodelsPoissonScoreModel:
    home_fit: Any
    away_fit: Any
    training_rows: int
    categories_home_team: tuple[str, ...]
    categories_away_team: tuple[str, ...]
    categories_competition: tuple[str, ...]
    method: str = "statsmodels_poisson_glm_score_v1"


def _prepare(history: pd.DataFrame, *, prediction_cutoff_utc: str | pd.Timestamp) -> pd.DataFrame:
    missing = sorted(_REQUIRED - set(history.columns))
    if missing:
        raise ValueError(f"statsmodels score training data missing columns: {missing}")

    d = history.copy()
    cutoff = pd.to_datetime(prediction_cutoff_utc, utc=True, errors="coerce")
    if pd.isna(cutoff):
        raise ValueError("prediction_cutoff_utc must be a valid timezone-aware timestamp")

    d["home_team"] = d["home_team"].astype(str).str.strip()
    d["away_team"] = d["away_team"].astype(str).str.strip()
    d["competition"] = d["competition"].astype(str).str.strip()
    d["kickoff_utc"] = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    d["source_available_at_utc"] = pd.to_datetime(
        d["source_available_at_utc"], utc=True, errors="coerce"
    )
    d["home_goals"] = pd.to_numeric(d["home_goals"], errors="coerce")
    d["away_goals"] = pd.to_numeric(d["away_goals"], errors="coerce")
    d["pit_verified"] = d["pit_verified"].astype("boolean")

    # Fail closed on unknown timing. Both the event and the predictor-side source
    # availability must precede the prediction cutoff.
    d = d[
        d["pit_verified"].eq(True)
        & d["kickoff_utc"].notna()
        & d["source_available_at_utc"].notna()
        & (d["kickoff_utc"] < cutoff)
        & (d["source_available_at_utc"] <= cutoff)
        & d["home_goals"].notna()
        & d["away_goals"].notna()
        & d["home_team"].ne("")
        & d["away_team"].ne("")
        & d["competition"].ne("")
    ].copy()

    if d.empty:
        raise ValueError("No PIT-verified rows available for statsmodels challenger")

    if (d[["home_goals", "away_goals"]] < 0).any().any():
        raise ValueError("Goal labels must be non-negative")

    return d.reset_index(drop=True)


def fit_statsmodels_poisson_score_model(
    history: pd.DataFrame,
    *,
    prediction_cutoff_utc: str | pd.Timestamp,
) -> StatsmodelsPoissonScoreModel:
    """Fit only on PIT-verified observations strictly before the locked cutoff."""
    d = _prepare(history, prediction_cutoff_utc=prediction_cutoff_utc)

    # Categorical terms make team/competition effects explicit while keeping the
    # challenger compact. Statsmodels handles the reference levels internally.
    formula_home = "home_goals ~ C(home_team) + C(away_team) + C(competition)"
    formula_away = "away_goals ~ C(home_team) + C(away_team) + C(competition)"

    home_fit = smf.glm(
        formula=formula_home,
        data=d,
        family=sm.families.Poisson(),
    ).fit()
    away_fit = smf.glm(
        formula=formula_away,
        data=d,
        family=sm.families.Poisson(),
    ).fit()

    return StatsmodelsPoissonScoreModel(
        home_fit=home_fit,
        away_fit=away_fit,
        training_rows=int(len(d)),
        categories_home_team=tuple(sorted(d["home_team"].unique())),
        categories_away_team=tuple(sorted(d["away_team"].unique())),
        categories_competition=tuple(sorted(d["competition"].unique())),
    )


def _check_categories(
    model: StatsmodelsPoissonScoreModel,
    home_team: str,
    away_team: str,
    competition: str,
) -> None:
    if str(home_team) not in model.categories_home_team:
        raise RuntimeError("statsmodels score model lacks PIT-trained home_team category")
    if str(away_team) not in model.categories_away_team:
        raise RuntimeError("statsmodels score model lacks PIT-trained away_team category")
    if str(competition) not in model.categories_competition:
        raise RuntimeError("statsmodels score model lacks PIT-trained competition category")


def _predict_mean(
    fit: Any,
    home_team: str,
    away_team: str,
    competition: str,
) -> float:
    frame = pd.DataFrame(
        {
            "home_team": [str(home_team)],
            "away_team": [str(away_team)],
            "competition": [str(competition)],
        }
    )
    value = float(np.asarray(fit.predict(frame), dtype=float).reshape(-1)[0])
    if not np.isfinite(value) or value <= 0:
        raise RuntimeError("statsmodels Poisson prediction is non-finite or non-positive")
    return value


def predict_statsmodels_poisson_distribution(
    model: StatsmodelsPoissonScoreModel,
    home_team: str,
    away_team: str,
    competition: str | None = None,
    *,
    max_goals: int = 12,
) -> list[tuple[int, int, float]]:
    if max_goals < 1:
        raise ValueError("max_goals must be at least 1")
    if competition is None or str(competition).strip() == "":
        raise RuntimeError("competition is required; unknown competition fails closed")
    competition_value = str(competition).strip()
    _check_categories(model, home_team, away_team, competition_value)

    home_lambda = _predict_mean(
        model.home_fit,
        home_team,
        away_team,
        competition_value,
    )
    away_lambda = _predict_mean(
        model.away_fit,
        home_team,
        away_team,
        competition_value,
    )

    distribution = []
    total = 0.0
    for home_goals in range(max_goals + 1):
        p_home = float(poisson.pmf(home_goals, home_lambda))
        for away_goals in range(max_goals + 1):
            p_away = float(poisson.pmf(away_goals, away_lambda))
            probability = p_home * p_away
            distribution.append((home_goals, away_goals, probability))
            total += probability

    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("statsmodels score distribution mass is invalid")

    # Truncation at max_goals removes a small tail. Renormalize so downstream
    # O/U/BTTS/Top-k metrics receive a proper probability distribution.
    normalized = [
        (h, a, float(p / total))
        for h, a, p in distribution
    ]
    if not np.isclose(sum(p for _, _, p in normalized), 1.0, atol=1e-9):
        raise RuntimeError("statsmodels score distribution failed normalization")
    return normalized
