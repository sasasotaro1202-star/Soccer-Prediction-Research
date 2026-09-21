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

    rows: dict[str, dict[str, float]] = {}
    teams = sorted(set(d["home_team"].astype(str)) | set(d["away_team"].astype(str)))
    for team in teams:
        home = d[d["home_team"].astype(str) == team]
        away = d[d["away_team"].astype(str) == team]

        home_n = len(home)
        away_n = len(away)
        home_scored = float(home["home_goals"].sum())
        home_conceded = float(home["away_goals"].sum())
        away_scored = float(away["away_goals"].sum())
        away_conceded = float(away["home_goals"].sum())

        # Venue-specific rates retain the important home/away asymmetry while
        # shrinkage keeps small-sample teams close to the global environment.
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
        "teams": rows,
    }


def predict_score_candidates(score_model: dict[str, Any], home_team: str, away_team: str) -> list[dict[str, Any]]:
    teams = score_model.get("teams", {})
    home = teams.get(str(home_team))
    away = teams.get(str(away_team))
    base_home = max(float(score_model["home_mean"]), 1e-6)
    base_away = max(float(score_model["away_mean"]), 1e-6)
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
    home_lambda = float(np.clip(home_lambda, 0.05, 5.0))
    away_lambda = float(np.clip(away_lambda, 0.05, 5.0))

    candidates = score_distribution(home_lambda, away_lambda, max_goals=7)
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
