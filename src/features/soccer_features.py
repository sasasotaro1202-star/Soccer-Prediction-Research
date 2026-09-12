from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from src.data.pit_policy import is_available_by_cutoff, result_feature_available_at

ELO_K = 20.0
ELO_HOME_ADV = 55.0


def _result_available(ts: pd.Timestamp, cutoff: pd.Timestamp) -> bool:
    return bool(is_available_by_cutoff(ts, cutoff))


def _team_result(row, team: str) -> tuple[float, float, int, str, float]:
    home = row["home_team"] == team
    gf = float(row["home_goals"] if home else row["away_goals"])
    ga = float(row["away_goals"] if home else row["home_goals"])
    pts = 3 if gf > ga else 1 if gf == ga else 0
    venue = "H" if home else "A"
    return gf, ga, pts, venue, gf - ga


def _ewma(values: list[float], alpha: float = 0.35) -> float:
    if not values:
        return np.nan
    out = float(values[0])
    for v in values[1:]:
        out = alpha * float(v) + (1.0 - alpha) * out
    return out


def _summarize(games: deque, window: int) -> dict[str, float]:
    recent = list(games)[-window:]
    if not recent:
        return {"games": 0.0, "gf": np.nan, "ga": np.nan, "points": np.nan, "gd": np.nan,
                "win_rate": np.nan, "draw_rate": np.nan, "loss_rate": np.nan,
                "gf_ewma": np.nan, "ga_ewma": np.nan, "gd_std": np.nan, "home_rate": np.nan}
    gf = [x["gf"] for x in recent]; ga = [x["ga"] for x in recent]
    pts = [x["points"] for x in recent]; gd = [x["gd"] for x in recent]
    return {
        "games": float(len(recent)), "gf": float(np.mean(gf)), "ga": float(np.mean(ga)),
        "points": float(np.mean(pts)), "gd": float(np.mean(gd)),
        "win_rate": float(np.mean([p == 3 for p in pts])),
        "draw_rate": float(np.mean([p == 1 for p in pts])),
        "loss_rate": float(np.mean([p == 0 for p in pts])),
        "gf_ewma": _ewma(gf), "ga_ewma": _ewma(ga),
        "gd_std": float(np.std(gd)) if len(gd) > 1 else 0.0,
        "home_rate": float(np.mean([x["venue"] == "H" for x in recent])),
    }


def _update_elo(elo: dict, home: str, away: str, result: int, competition: str) -> None:
    actual = 1.0 if result == 0 else 0.5 if result == 1 else 0.0
    he = float(elo["global"].get(home, 1500.0)); ae = float(elo["global"].get(away, 1500.0))
    expected = 1.0 / (1.0 + 10.0 ** (-(he + ELO_HOME_ADV - ae) / 400.0))
    delta = ELO_K * (actual - expected)
    elo["global"][home] = he + delta; elo["global"][away] = ae - delta
    ce = elo["competition"].setdefault(competition, {})
    che = float(ce.get(home, 1500.0)); cae = float(ce.get(away, 1500.0))
    expected_c = 1.0 / (1.0 + 10.0 ** (-(che + ELO_HOME_ADV - cae) / 400.0))
    delta_c = ELO_K * (actual - expected_c)
    ce[home] = che + delta_c; ce[away] = cae - delta_c


def build_match_features(history: pd.DataFrame, matches: pd.DataFrame, windows=(3, 5, 10, 20)) -> pd.DataFrame:
    """Build chronological, leakage-safe features using explicit result availability.

    If a source supplies source_available_at_utc, it is authoritative. Otherwise the
    conservative result-availability policy is used. No unavailable result can enter
    the rolling state. The implementation is deliberately single-pass for speed.
    """
    h = history.copy()
    h["kickoff_utc"] = pd.to_datetime(h["kickoff_utc"], utc=True, errors="coerce")
    if "source_available_at_utc" in h.columns:
        h["source_available_at_utc"] = pd.to_datetime(h["source_available_at_utc"], utc=True, errors="coerce")
    h = h.dropna(subset=["kickoff_utc"]).sort_values(
        ["kickoff_utc", "competition", "home_team", "away_team", "match_id"], kind="mergesort"
    ).reset_index(drop=True)
    m = matches.copy()
    m["kickoff_utc"] = pd.to_datetime(m["kickoff_utc"], utc=True, errors="coerce")
    m = m.dropna(subset=["kickoff_utc"]).sort_values(
        ["kickoff_utc", "competition", "home_team", "away_team", "match_id"], kind="mergesort"
    ).reset_index(drop=True)

    cols = ["kickoff_utc", "home_team", "away_team", "home_goals", "away_goals", "competition", "match_id"]
    if "source_available_at_utc" in h.columns:
        cols.append("source_available_at_utc")
    h_records = h[cols].to_dict("records")
    m_records = m[[c for c in ["match_id", "competition", "season", "season_start", "kickoff_utc", "home_team", "away_team"] if c in m.columns]].to_dict("records")

    team_games: dict[str, deque] = defaultdict(lambda: deque(maxlen=40))
    team_last: dict[str, pd.Timestamp] = {}
    team_last_available: dict[str, pd.Timestamp] = {}
    h2h: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=10))
    elo = {"global": {}, "competition": {}}
    ptr = 0; rows = []

    def ingest_until(cutoff: pd.Timestamp) -> None:
        nonlocal ptr
        while ptr < len(h_records):
            r = h_records[ptr]
            event = r["kickoff_utc"]
            # We require both the event to have happened and the result to be available.
            source_available = r.get("source_available_at_utc")
            available = source_available if pd.notna(source_available) else result_feature_available_at(event)
            if event >= cutoff or pd.isna(available) or available > cutoff:
                break
            hg, ag = r["home_goals"], r["away_goals"]
            if pd.isna(hg) or pd.isna(ag):
                ptr += 1
                continue
            home, away = str(r["home_team"]), str(r["away_team"])
            result = 0 if hg > ag else 1 if hg == ag else 2
            _update_elo(elo, home, away, result, str(r["competition"]))
            for team in (home, away):
                gf, ga, pts, venue, gd = _team_result(r, team)
                team_games[team].append({"time": event, "gf": gf, "ga": ga, "points": pts, "venue": venue, "gd": gd, "competition": str(r["competition"])})
                team_last[team] = event
                team_last_available[team] = available
            h2h[(home, away)].append(result)
            h2h[(away, home)].append(2 - result if result != 1 else 1)
            ptr += 1

    for r in m_records:
        kickoff = r["kickoff_utc"]
        cutoff = kickoff - pd.Timedelta(minutes=60)
        ingest_until(cutoff)
        home, away, comp = str(r["home_team"]), str(r["away_team"]), str(r["competition"])
        he = float(elo["global"].get(home, 1500.0)); ae = float(elo["global"].get(away, 1500.0))
        ce = elo["competition"].get(comp, {}); hce = float(ce.get(home, 1500.0)); cae = float(ce.get(away, 1500.0))
        row = {
            "match_id": r["match_id"], "competition": comp, "season": r.get("season"), "season_start": r.get("season_start", np.nan),
            "kickoff_utc": kickoff, "home_team": home, "away_team": away, "prediction_cutoff_at_utc": cutoff,
            "home_advantage": 1.0, "home_elo": he, "away_elo": ae, "elo_diff": he - ae,
            "home_comp_elo": hce, "away_comp_elo": cae, "comp_elo_diff": hce - cae,
            "home_elo_expected": 1.0 / (1.0 + 10.0 ** (-((he + ELO_HOME_ADV) - ae) / 400.0)),
            "home_rest_hours": (cutoff - team_last[home]).total_seconds() / 3600.0 if home in team_last else np.nan,
            "away_rest_hours": (cutoff - team_last[away]).total_seconds() / 3600.0 if away in team_last else np.nan,
        }
        for w in windows:
            hs = _summarize(team_games[home], w); aws = _summarize(team_games[away], w)
            for k, v in hs.items(): row[f"home_{k}_{w}"] = v
            for k, v in aws.items(): row[f"away_{k}_{w}"] = v
            row[f"gf_diff_{w}"] = hs["gf"] - aws["gf"]
            row[f"ga_diff_{w}"] = hs["ga"] - aws["ga"]
            row[f"points_diff_{w}"] = hs["points"] - aws["points"]
            row[f"win_rate_diff_{w}"] = hs["win_rate"] - aws["win_rate"]
        meetings = list(h2h[(home, away)])[-5:]
        row["h2h_games_5"] = float(len(meetings))
        row["h2h_home_win_rate_5"] = float(np.mean([x == 0 for x in meetings])) if meetings else np.nan
        row["h2h_draw_rate_5"] = float(np.mean([x == 1 for x in meetings])) if meetings else np.nan
        row["h2h_away_win_rate_5"] = float(np.mean([x == 2 for x in meetings])) if meetings else np.nan
        row["h2h_points_edge_5"] = float(np.mean([3 if x == 0 else 1 if x == 1 else 0 for x in meetings]) - np.mean([3 if x == 2 else 1 if x == 1 else 0 for x in meetings])) if meetings else np.nan
        available_times = [team_last_available[t] for t in (home, away) if t in team_last_available]
        row["feature_source_max_available_at_utc"] = max(available_times) if available_times else pd.NaT
        row["pit_verified"] = all(t in team_last and team_last_available.get(t, pd.NaT) <= cutoff for t in (home, away))
        rows.append(row)
    return pd.DataFrame(rows)


def add_target(features: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    actual = matches[["match_id", "home_goals", "away_goals"]].copy()
    out = features.merge(actual, on="match_id", how="left", validate="one_to_one")
    out["target"] = np.where(out.home_goals > out.away_goals, 0, np.where(out.home_goals == out.away_goals, 1, 2))
    return out
