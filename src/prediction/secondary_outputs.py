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

    rows: dict[str, dict[str, float]] = {}
    teams = sorted(set(d["home_team"].astype(str)) | set(d["away_team"].astype(str)))
    for team in teams:
        home = d[d["home_team"].astype(str) == team]
        away = d[d["away_team"].astype(str) == team]
        n = len(home) + len(away)
        scored = float(home["home_goals"].sum() + away["away_goals"].sum())
        conceded = float(home["away_goals"].sum() + away["home_goals"].sum())
        scored_rate = (scored + shrinkage * ((home_mean + away_mean) / 2.0)) / (n + shrinkage)
        conceded_rate = (conceded + shrinkage * ((home_mean + away_mean) / 2.0)) / (n + shrinkage)
        rows[team] = {"scored_rate": scored_rate, "conceded_rate": conceded_rate}

    return {
        "schema_version": 1,
        "method": "pit_smoothed_team_goal_rates",
        "shrinkage": float(shrinkage),
        "home_mean": home_mean,
        "away_mean": away_mean,
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

    league_avg = max((base_home + base_away) / 2.0, 1e-6)
    home_lambda = base_home * (float(home["scored_rate"]) / league_avg) * (float(away["conceded_rate"]) / league_avg)
    away_lambda = base_away * (float(away["scored_rate"]) / league_avg) * (float(home["conceded_rate"]) / league_avg)
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
