"""Production-safe secondary prediction outputs.

Score uses a PIT-trained team-rate Poisson model stored in the production bundle.
MOM is intentionally fail-closed: player candidates and their PIT-safe probabilities
must be supplied by the upstream player model; this module never fabricates players
or probabilities.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.score import score_distribution
from src.models.mom_selector import select_mom_candidates
from src.models.score_selector import select_score_candidates


def fit_score_rate_model(history: pd.DataFrame, *, shrinkage: float = 20.0) -> dict[str, Any]:
    required = {"home_team", "away_team", "home_goals", "away_goals", "pit_verified"}
    missing = sorted(required - set(history.columns))
    if missing:
        raise ValueError(f"Score-rate training data missing columns: {missing}")
    d = history[history["pit_verified"] == True].copy()
    d["home_goals"] = pd.to_numeric(d["home_goals"], errors="coerce")
    d["away_goals"] = pd.to_numeric(d["away_goals"], errors="coerce")
    d = d.dropna(subset=["home_goals", "away_goals", "home_team", "away_team"])
    if d.empty:
        raise ValueError("No PIT-verified score rows available")
    home_mean = float(d["home_goals"].mean())
    away_mean = float(d["away_goals"].mean())
    overall_mean = float((d["home_goals"].sum() + d["away_goals"].sum()) / max(1, 2 * len(d)))

    # Competition-specific scoring environments are shrunk toward the global venue means.
    # This separates league/run-environment effects from team strength without introducing
    # high-dimensional competition x team parameters.
    competition_rates: dict[str, dict[str, float]] = {}
    if "competition" in d.columns:
        comp = d.assign(_competition=d["competition"].astype(str)).groupby("_competition", sort=True)
        for name, g in comp:
            n = len(g)
            competition_rates[str(name)] = {
                "home_mean": float((g["home_goals"].sum() + shrinkage * home_mean) / (n + shrinkage)),
                "away_mean": float((g["away_goals"].sum() + shrinkage * away_mean) / (n + shrinkage)),
                "matches": float(n),
            }

    rows: dict[str, dict[str, float]] = {}
    d_home = d.assign(_team=d["home_team"].astype(str))
    d_away = d.assign(_team=d["away_team"].astype(str))
    home_agg = d_home.groupby("_team", sort=False).agg(
        home_matches=("home_goals", "size"),
        home_scored=("home_goals", "sum"),
        home_conceded=("away_goals", "sum"),
    )
    away_agg = d_away.groupby("_team", sort=False).agg(
        away_matches=("away_goals", "size"),
        away_scored=("away_goals", "sum"),
        away_conceded=("home_goals", "sum"),
    )
    teams = sorted(set(home_agg.index.astype(str)) | set(away_agg.index.astype(str)))
    for team in teams:
        h = home_agg.loc[team] if team in home_agg.index else None
        a = away_agg.loc[team] if team in away_agg.index else None
        home_n = int(h["home_matches"]) if h is not None else 0
        away_n = int(a["away_matches"]) if a is not None else 0
        home_scored = float(h["home_scored"]) if h is not None else 0.0
        home_conceded = float(h["home_conceded"]) if h is not None else 0.0
        away_scored = float(a["away_scored"]) if a is not None else 0.0
        away_conceded = float(a["away_conceded"]) if a is not None else 0.0

        # Venue-specific rates retain home/away asymmetry while shrinkage keeps
        # small-sample teams close to the global scoring environment.
        home_scored_rate = (home_scored + shrinkage * home_mean) / (home_n + shrinkage)
        home_conceded_rate = (home_conceded + shrinkage * away_mean) / (home_n + shrinkage)
        away_scored_rate = (away_scored + shrinkage * away_mean) / (away_n + shrinkage)
        away_conceded_rate = (away_conceded + shrinkage * home_mean) / (away_n + shrinkage)

        all_n = home_n + away_n
        scored = home_scored + away_scored
        conceded = home_conceded + away_conceded
        scored_rate = (scored + shrinkage * overall_mean) / (all_n + shrinkage)
        conceded_rate = (conceded + shrinkage * overall_mean) / (all_n + shrinkage)

        rows[team] = {
            "home_scored_rate": home_scored_rate,
            "home_conceded_rate": home_conceded_rate,
            "away_scored_rate": away_scored_rate,
            "away_conceded_rate": away_conceded_rate,
            "scored_rate": scored_rate,
            "conceded_rate": conceded_rate,
            "home_matches": float(home_n),
            "away_matches": float(away_n),
        }

    return {
        "schema_version": 2,
        "method": "pit_smoothed_venue_split_team_goal_rates",
        "shrinkage": float(shrinkage),
        "home_mean": home_mean,
        "away_mean": away_mean,
        "overall_mean": overall_mean,
        "competition_rates": competition_rates,
        "teams": rows,
    }



def fit_recency_score_rate_model(
    history: pd.DataFrame,
    *,
    shrinkage: float = 20.0,
    half_life_rows: float = 800.0,
) -> dict[str, Any]:
    """PIT-safe score-rate challenger with deterministic exponential row decay."""
    required = {"kickoff_utc", "home_team", "away_team", "home_goals", "away_goals", "pit_verified"}
    missing = sorted(required - set(history.columns))
    if missing:
        raise ValueError(f"Recency score training data missing columns: {missing}")
    d = history.loc[history["pit_verified"] == True].copy()
    d["kickoff_utc"] = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    d["home_goals"] = pd.to_numeric(d["home_goals"], errors="coerce")
    d["away_goals"] = pd.to_numeric(d["away_goals"], errors="coerce")
    d = d.dropna(subset=["kickoff_utc", "home_goals", "away_goals", "home_team", "away_team"])
    d = d.sort_values("kickoff_utc", kind="mergesort").reset_index(drop=True)
    if d.empty:
        raise ValueError("No PIT-verified score rows available")
    half = max(float(half_life_rows), 1.0)
    pos = np.arange(len(d), dtype=float)
    d["_weight"] = np.exp((pos - float(len(d) - 1)) / half)
    weight_sum = max(float(d["_weight"].sum()), 1e-12)
    home_mean = float((d["home_goals"] * d["_weight"]).sum() / weight_sum)
    away_mean = float((d["away_goals"] * d["_weight"]).sum() / weight_sum)
    overall_mean = float(((d["home_goals"] + d["away_goals"]) * d["_weight"]).sum() / (2.0 * weight_sum))

    competition_rates: dict[str, dict[str, float]] = {}
    if "competition" in d.columns:
        comp = d.assign(
            _competition=d["competition"].astype(str),
            _whg=d["home_goals"] * d["_weight"],
            _wag=d["away_goals"] * d["_weight"],
        )
        for name, g in comp.groupby("_competition", sort=True):
            w = float(g["_weight"].sum())
            competition_rates[str(name)] = {
                "home_mean": float((g["_whg"].sum() + shrinkage * home_mean) / (w + shrinkage)),
                "away_mean": float((g["_wag"].sum() + shrinkage * away_mean) / (w + shrinkage)),
                "matches": float(w),
            }

    d["_home_team"] = d["home_team"].astype(str)
    d["_away_team"] = d["away_team"].astype(str)
    d["_home_scored"] = d["home_goals"] * d["_weight"]
    d["_home_conceded"] = d["away_goals"] * d["_weight"]
    d["_away_scored"] = d["away_goals"] * d["_weight"]
    d["_away_conceded"] = d["home_goals"] * d["_weight"]
    home_agg = d.groupby("_home_team", sort=False)[["_weight", "_home_scored", "_home_conceded"]].sum()
    away_agg = d.groupby("_away_team", sort=False)[["_weight", "_away_scored", "_away_conceded"]].sum()
    home_agg = home_agg.rename(columns={"_weight": "home_weight"})
    away_agg = away_agg.rename(columns={"_weight": "away_weight"})
    teams = sorted(set(home_agg.index.astype(str)) | set(away_agg.index.astype(str)))

    rows: dict[str, dict[str, float]] = {}
    for team in teams:
        h = home_agg.loc[team] if team in home_agg.index else None
        a = away_agg.loc[team] if team in away_agg.index else None
        home_w = float(h["home_weight"]) if h is not None else 0.0
        away_w = float(a["away_weight"]) if a is not None else 0.0
        home_scored = float(h["_home_scored"]) if h is not None else 0.0
        home_conceded = float(h["_home_conceded"]) if h is not None else 0.0
        away_scored = float(a["_away_scored"]) if a is not None else 0.0
        away_conceded = float(a["_away_conceded"]) if a is not None else 0.0
        home_scored_rate = (home_scored + shrinkage * home_mean) / (home_w + shrinkage)
        home_conceded_rate = (home_conceded + shrinkage * away_mean) / (home_w + shrinkage)
        away_scored_rate = (away_scored + shrinkage * away_mean) / (away_w + shrinkage)
        away_conceded_rate = (away_conceded + shrinkage * home_mean) / (away_w + shrinkage)
        all_w = home_w + away_w
        rows[team] = {
            "home_scored_rate": home_scored_rate,
            "home_conceded_rate": home_conceded_rate,
            "away_scored_rate": away_scored_rate,
            "away_conceded_rate": away_conceded_rate,
            "scored_rate": (home_scored + away_scored + shrinkage * overall_mean) / (all_w + shrinkage),
            "conceded_rate": (home_conceded + away_conceded + shrinkage * overall_mean) / (all_w + shrinkage),
            "home_matches": home_w,
            "away_matches": away_w,
        }
    return {
        "schema_version": 1,
        "method": "pit_recency_weighted_venue_split_team_goal_rates",
        "training_rows": int(len(d)),
        "shrinkage": float(shrinkage),
        "half_life_rows": float(half),
        "effective_weight_sum": float(weight_sum),
        "home_mean": home_mean,
        "away_mean": away_mean,
        "overall_mean": overall_mean,
        "competition_rates": competition_rates,
        "teams": rows,
    }


def fit_time_decay_score_rate_model(
    history: pd.DataFrame,
    *,
    shrinkage: float = 20.0,
    half_life_days: float = 180.0,
) -> dict[str, Any]:
    """PIT-safe score-rate challenger with deterministic exponential time decay."""
    required = {"kickoff_utc", "home_team", "away_team", "home_goals", "away_goals", "pit_verified"}
    missing = sorted(required - set(history.columns))
    if missing:
        raise ValueError(f"Time-decay score training data missing columns: {missing}")
    d = history.loc[history["pit_verified"] == True].copy()
    d["kickoff_utc"] = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    d["home_goals"] = pd.to_numeric(d["home_goals"], errors="coerce")
    d["away_goals"] = pd.to_numeric(d["away_goals"], errors="coerce")
    d = d.dropna(subset=["kickoff_utc", "home_goals", "away_goals", "home_team", "away_team"])
    d = d.sort_values("kickoff_utc", kind="mergesort").reset_index(drop=True)
    if d.empty:
        raise ValueError("No PIT-verified score rows available")
    half_days = max(float(half_life_days), 1.0)
    latest = d["kickoff_utc"].max()
    age_days = (latest - d["kickoff_utc"]).dt.total_seconds() / 86400.0
    d["_weight"] = np.exp(-np.log(2.0) * age_days / half_days)
    weight_sum = max(float(d["_weight"].sum()), 1e-12)
    home_mean = float((d["home_goals"] * d["_weight"]).sum() / weight_sum)
    away_mean = float((d["away_goals"] * d["_weight"]).sum() / weight_sum)
    overall_mean = float(((d["home_goals"] + d["away_goals"]) * d["_weight"]).sum() / (2.0 * weight_sum))

    competition_rates: dict[str, dict[str, float]] = {}
    if "competition" in d.columns:
        comp = d.assign(
            _competition=d["competition"].astype(str),
            _whg=d["home_goals"] * d["_weight"],
            _wag=d["away_goals"] * d["_weight"],
        )
        for name, g in comp.groupby("_competition", sort=True):
            w = float(g["_weight"].sum())
            competition_rates[str(name)] = {
                "home_mean": float((g["_whg"].sum() + shrinkage * home_mean) / (w + shrinkage)),
                "away_mean": float((g["_wag"].sum() + shrinkage * away_mean) / (w + shrinkage)),
                "matches": float(w),
            }

    d["_home_team"] = d["home_team"].astype(str)
    d["_away_team"] = d["away_team"].astype(str)
    d["_home_scored"] = d["home_goals"] * d["_weight"]
    d["_home_conceded"] = d["away_goals"] * d["_weight"]
    d["_away_scored"] = d["away_goals"] * d["_weight"]
    d["_away_conceded"] = d["home_goals"] * d["_weight"]
    home_agg = d.groupby("_home_team", sort=False)[["_weight", "_home_scored", "_home_conceded"]].sum()
    away_agg = d.groupby("_away_team", sort=False)[["_weight", "_away_scored", "_away_conceded"]].sum()
    home_agg = home_agg.rename(columns={"_weight": "home_weight"})
    away_agg = away_agg.rename(columns={"_weight": "away_weight"})
    teams = sorted(set(home_agg.index.astype(str)) | set(away_agg.index.astype(str)))

    rows: dict[str, dict[str, float]] = {}
    for team in teams:
        h = home_agg.loc[team] if team in home_agg.index else None
        a = away_agg.loc[team] if team in away_agg.index else None
        home_w = float(h["home_weight"]) if h is not None else 0.0
        away_w = float(a["away_weight"]) if a is not None else 0.0
        home_scored = float(h["_home_scored"]) if h is not None else 0.0
        home_conceded = float(h["_home_conceded"]) if h is not None else 0.0
        away_scored = float(a["_away_scored"]) if a is not None else 0.0
        away_conceded = float(a["_away_conceded"]) if a is not None else 0.0
        home_scored_rate = (home_scored + shrinkage * home_mean) / (home_w + shrinkage)
        home_conceded_rate = (home_conceded + shrinkage * away_mean) / (home_w + shrinkage)
        away_scored_rate = (away_scored + shrinkage * away_mean) / (away_w + shrinkage)
        away_conceded_rate = (away_conceded + shrinkage * home_mean) / (away_w + shrinkage)
        all_w = home_w + away_w
        rows[team] = {
            "home_scored_rate": home_scored_rate,
            "home_conceded_rate": home_conceded_rate,
            "away_scored_rate": away_scored_rate,
            "away_conceded_rate": away_conceded_rate,
            "scored_rate": (home_scored + away_scored + shrinkage * overall_mean) / (all_w + shrinkage),
            "conceded_rate": (home_conceded + away_conceded + shrinkage * overall_mean) / (all_w + shrinkage),
            "home_matches": home_w,
            "away_matches": away_w,
        }
    return {
        "schema_version": 1,
        "method": "pit_time_decay_weighted_venue_split_team_goal_rates",
        "training_rows": int(len(d)),
        "shrinkage": float(shrinkage),
        "half_life_days": float(half_days),
        "effective_weight_sum": float(weight_sum),
        "home_mean": home_mean,
        "away_mean": away_mean,
        "overall_mean": overall_mean,
        "competition_rates": competition_rates,
        "teams": rows,
    }

def _score_lambdas(
    score_model: dict[str, Any],
    home_team: str,
    away_team: str,
    competition: str | None = None,
) -> tuple[float, float]:
    teams = score_model.get("teams", {})
    home = teams.get(str(home_team))
    away = teams.get(str(away_team))
    base_home = max(float(score_model["home_mean"]), 1e-6)
    base_away = max(float(score_model["away_mean"]), 1e-6)
    comp_rates = score_model.get("competition_rates", {})
    comp = comp_rates.get(str(competition)) if competition is not None else None
    if isinstance(comp, dict):
        base_home = max(float(comp.get("home_mean", base_home)), 1e-6)
        base_away = max(float(comp.get("away_mean", base_away)), 1e-6)
    if home is None or away is None:
        raise RuntimeError("Score model has no PIT-trained rate for one or both fixture teams")

    # Venue-split formulation: attack rate and opponent defensive concession rate
    # are combined around their matching global venue mean.
    home_attack = float(home.get("home_scored_rate", home["scored_rate"]))
    home_defense = float(home.get("home_conceded_rate", home["conceded_rate"]))
    away_attack = float(away.get("away_scored_rate", away["scored_rate"]))
    away_defense = float(away.get("away_conceded_rate", away["conceded_rate"]))
    league_home = max(base_home, 1e-6)
    league_away = max(base_away, 1e-6)
    home_lambda = league_home * (home_attack / league_home) * (away_defense / league_home)
    away_lambda = league_away * (away_attack / league_away) * (home_defense / league_away)
    return float(np.clip(home_lambda, 0.05, 5.0)), float(np.clip(away_lambda, 0.05, 5.0))


def predict_score_distribution(
    score_model: dict[str, Any],
    home_team: str,
    away_team: str,
    competition: str | None = None,
    *,
    max_goals: int = 12,
) -> list[tuple[int, int, float]]:
    """Return the full PIT-trained score distribution, dispatching by locked method."""
    if max_goals < 1:
        raise ValueError("max_goals must be at least 1")
    method = str(score_model.get("method", ""))
    if method.startswith("dixon_coles_"):
        from src.models.dixon_coles import predict_dixon_coles_distribution
        return predict_dixon_coles_distribution(
            score_model,
            home_team,
            away_team,
            competition,
            max_goals=max_goals,
        )
    home_lambda, away_lambda = _score_lambdas(score_model, home_team, away_team, competition)
    return score_distribution(home_lambda, away_lambda, max_goals=max_goals)


def predict_score_candidates(
    score_model: dict[str, Any],
    home_team: str,
    away_team: str,
    competition: str | None = None,
) -> list[dict[str, Any]]:
    candidates = predict_score_distribution(score_model, home_team, away_team, competition, max_goals=12)
    selected = select_score_candidates(candidates)
    return [
        {
            "home_goals": x.home_goals,
            "away_goals": x.away_goals,
            "probability": x.probability,
            "rank": x.rank,
        }
        for x in selected
    ]



def predict_score_markets(
    score_model: dict[str, Any],
    home_team: str,
    away_team: str,
    competition: str | None = None,
) -> dict[str, float]:
    """Return O/U and BTTS probabilities from the same full score distribution."""
    distribution = predict_score_distribution(score_model, home_team, away_team, competition, max_goals=12)
    total = np.asarray([h + a for h, a, _ in distribution], dtype=float)
    home_goals = np.asarray([h for h, _, _ in distribution], dtype=int)
    away_goals = np.asarray([a for _, a, _ in distribution], dtype=int)
    probs = np.asarray([p for _, _, p in distribution], dtype=float)
    out: dict[str, float] = {}
    for line in (0.5, 1.5, 2.5, 3.5, 4.5):
        key = str(line).replace(".5", "_5")
        out[f"over_{key}"] = float(probs[total > line].sum())
        out[f"under_{key}"] = float(probs[total < line].sum())
    out["btts_yes"] = float(probs[(home_goals >= 1) & (away_goals >= 1)].sum())
    out["btts_no"] = float(probs[(home_goals == 0) | (away_goals == 0)].sum())
    return out

def _parse_mom_input(value: Any) -> tuple[list[str], list[float]]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("MOM input must be a JSON list")
    ids, probs = [], []
    for row in value:
        if not isinstance(row, dict) or "player_id" not in row or "probability" not in row:
            raise ValueError("Each MOM candidate must contain player_id and probability")
        ids.append(str(row["player_id"]))
        probs.append(float(row["probability"]))
    return ids, probs


def predict_mom_candidates(player_candidates: Any) -> list[dict[str, Any]]:
    ids, probs = _parse_mom_input(player_candidates)
    selected = select_mom_candidates(ids, probs)
    return [asdict(x) for x in selected]
