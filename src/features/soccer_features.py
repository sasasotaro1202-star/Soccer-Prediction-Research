from __future__ import annotations

import numpy as np
import pandas as pd


def _state(history: pd.DataFrame, team: str, cutoff: pd.Timestamp, window: int = 5) -> dict[str, float]:
    h = history[(history.home_team == team) | (history.away_team == team)].copy()
    h = h[h.kickoff_utc < cutoff].sort_values("kickoff_utc").tail(window)
    if h.empty:
        return {"games": 0, "gf": np.nan, "ga": np.nan, "points": np.nan, "gd": np.nan}
    gf, ga, pts = [], [], []
    for r in h.itertuples():
        home = r.home_team == team
        f = r.home_goals if home else r.away_goals
        a = r.away_goals if home else r.home_goals
        gf.append(float(f)); ga.append(float(a))
        pts.append(3 if f > a else 1 if f == a else 0)
    return {"games": len(h), "gf": np.mean(gf), "ga": np.mean(ga), "points": np.mean(pts), "gd": np.mean(np.array(gf)-np.array(ga))}


def build_match_features(history: pd.DataFrame, matches: pd.DataFrame, windows=(3, 5, 10)) -> pd.DataFrame:
    history = history.copy().sort_values("kickoff_utc")
    rows = []
    for r in matches.sort_values("kickoff_utc").itertuples():
        row = {"match_id": r.match_id, "competition": r.competition, "season": r.season,
               "kickoff_utc": r.kickoff_utc, "home_team": r.home_team, "away_team": r.away_team}
        prior = history[history.kickoff_utc < r.kickoff_utc]
        for w in windows:
            hs = _state(prior, r.home_team, r.kickoff_utc, w)
            aws = _state(prior, r.away_team, r.kickoff_utc, w)
            for k, v in hs.items(): row[f"home_{k}_{w}"] = v
            for k, v in aws.items(): row[f"away_{k}_{w}"] = v
        row["home_gd_5_minus_away_gd_5"] = row["home_gd_5"] - row["away_gd_5"]
        row["home_points_5_minus_away_points_5"] = row["home_points_5"] - row["away_points_5"]
        row["home_advantage"] = 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def add_target(features: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    actual = matches[["match_id", "home_goals", "away_goals"]].copy()
    out = features.merge(actual, on="match_id", how="left", validate="one_to_one")
    out["target"] = np.where(out.home_goals > out.away_goals, 0, np.where(out.home_goals == out.away_goals, 1, 2))
    return out
