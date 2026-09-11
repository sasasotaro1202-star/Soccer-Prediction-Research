from __future__ import annotations

import numpy as np
import pandas as pd


def _state(history: pd.DataFrame, team: str, cutoff: pd.Timestamp, window: int = 5) -> dict[str, float]:
    h = history[(history.home_team == team) | (history.away_team == team)].copy()
    h = h[h.kickoff_utc < cutoff].sort_values("kickoff_utc").tail(window)
    base = {"games": 0, "gf": np.nan, "ga": np.nan, "points": np.nan, "gd": np.nan, "pit_ok": 0.0}
    if len(h) < window:
        return base

    availability = pd.to_datetime(h.get("source_available_at_utc"), utc=True, errors="coerce")
    if availability.isna().any() or (availability > cutoff).any():
        return base

    gf, ga, pts = [], [], []
    for r in h.itertuples():
        home = r.home_team == team
        f = r.home_goals if home else r.away_goals
        a = r.away_goals if home else r.home_goals
        gf.append(float(f))
        ga.append(float(a))
        pts.append(3 if f > a else 1 if f == a else 0)
    return {
        "games": len(h),
        "gf": np.mean(gf),
        "ga": np.mean(ga),
        "points": np.mean(pts),
        "gd": np.mean(np.array(gf) - np.array(ga)),
        "pit_ok": 1.0,
    }


def build_match_features(history: pd.DataFrame, matches: pd.DataFrame, windows=(3, 5, 10)) -> pd.DataFrame:
    """Build features using only past results whose source was available by cutoff."""
    history = history.copy().sort_values("kickoff_utc")
    rows = []
    for r in matches.sort_values("kickoff_utc").itertuples():
        cutoff = pd.Timestamp(r.kickoff_utc) - pd.Timedelta(minutes=60)
        row = {
            "match_id": r.match_id,
            "competition": r.competition,
            "season": r.season,
            "kickoff_utc": r.kickoff_utc,
            "home_team": r.home_team,
            "away_team": r.away_team,
            "prediction_cutoff_at_utc": cutoff,
        }
        prior = history[history.kickoff_utc < r.kickoff_utc]
        pit_flags = []
        source_times = []
        for w in windows:
            hs = _state(prior, r.home_team, r.kickoff_utc, w)
            aws = _state(prior, r.away_team, r.kickoff_utc, w)
            for k, v in hs.items():
                row[f"home_{k}_{w}"] = v
            for k, v in aws.items():
                row[f"away_{k}_{w}"] = v
            pit_flags.extend([hs["pit_ok"], aws["pit_ok"]])

            for team in (r.home_team, r.away_team):
                team_rows = prior[(prior.home_team == team) | (prior.away_team == team)].sort_values("kickoff_utc").tail(w)
                if len(team_rows) == w:
                    times = pd.to_datetime(team_rows.get("source_available_at_utc"), utc=True, errors="coerce")
                    if not times.isna().any():
                        source_times.append(times.max())

        row["home_gd_5_minus_away_gd_5"] = row["home_gd_5"] - row["away_gd_5"]
        row["home_points_5_minus_away_points_5"] = row["home_points_5"] - row["away_points_5"]
        row["home_advantage"] = 1.0
        row["feature_source_max_available_at_utc"] = max(source_times) if source_times else pd.NaT
        row["pit_verified"] = bool(pit_flags) and all(flag == 1.0 for flag in pit_flags)
        rows.append(row)
    return pd.DataFrame(rows)


def add_target(features: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    actual = matches[["match_id", "home_goals", "away_goals"]].copy()
    out = features.merge(actual, on="match_id", how="left", validate="one_to_one")
    out["target"] = np.where(out.home_goals > out.away_goals, 0, np.where(out.home_goals == out.away_goals, 1, 2))
    return out
