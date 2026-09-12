from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from src.data.pit_policy import is_available_by_cutoff, result_feature_available_at

ELO_K = 20.0
ELO_HOME_ADV = 55.0
STAT_KEYS = ("shots", "shots_on_target", "corners", "fouls", "yellow_cards", "red_cards")


def _result_available(ts: pd.Timestamp, cutoff: pd.Timestamp) -> bool:
    return bool(is_available_by_cutoff(ts, cutoff))


def _team_result(row, team: str) -> tuple[float, float, int, str, float]:
    home = row["home_team"] == team
    gf = float(row["home_goals"] if home else row["away_goals"])
    ga = float(row["away_goals"] if home else row["home_goals"])
    pts = 3 if gf > ga else 1 if gf == ga else 0
    venue = "H" if home else "A"
    return gf, ga, pts, venue, gf - ga


def _stat_value(row: dict, team: str, key: str) -> float:
    prefix = "home_" if row["home_team"] == team else "away_"
    value = row.get(f"{prefix}{key}", np.nan)
    return float(value) if pd.notna(value) else np.nan


def _ewma(values: list[float], alpha: float = 0.35) -> float:
    values = [float(v) for v in values if pd.notna(v)]
    if not values:
        return np.nan
    out = float(values[0])
    for v in values[1:]:
        out = alpha * float(v) + (1.0 - alpha) * out
    return out


def _summarize(games: deque, window: int) -> dict[str, float]:
    recent = list(games)[-window:]
    if not recent:
        base = {"games": 0.0, "gf": np.nan, "ga": np.nan, "points": np.nan, "gd": np.nan,
                "win_rate": np.nan, "draw_rate": np.nan, "loss_rate": np.nan,
                "gf_ewma": np.nan, "ga_ewma": np.nan, "gd_std": np.nan, "home_rate": np.nan}
        base.update({f"{k}_avg": np.nan for k in STAT_KEYS})
        base.update({f"{k}_ewma": np.nan for k in STAT_KEYS})
        return base
    gf = [x["gf"] for x in recent]
    ga = [x["ga"] for x in recent]
    pts = [x["points"] for x in recent]
    gd = [x["gd"] for x in recent]
    out = {
        "games": float(len(recent)), "gf": float(np.mean(gf)), "ga": float(np.mean(ga)),
        "points": float(np.mean(pts)), "gd": float(np.mean(gd)),
        "win_rate": float(np.mean([p == 3 for p in pts])),
        "draw_rate": float(np.mean([p == 1 for p in pts])),
        "loss_rate": float(np.mean([p == 0 for p in pts])),
        "gf_ewma": _ewma(gf), "ga_ewma": _ewma(ga),
        "gd_std": float(np.std(gd)) if len(gd) > 1 else 0.0,
        "home_rate": float(np.mean([x["venue"] == "H" for x in recent])),
    }
    for k in STAT_KEYS:
        vals = [x.get(k, np.nan) for x in recent]
        out[f"{k}_avg"] = float(np.nanmean(vals)) if any(pd.notna(v) for v in vals) else np.nan
        out[f"{k}_ewma"] = _ewma(vals)
    return out


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
    """Build chronological PIT-safe features with linear-time history processing.

    A history row is processed only when both its event time and availability time
    are before the prediction cutoff. Crucially, an event that is temporarily not
    available is *not discarded*: the history pointer stays there and the event can
    enter a later cutoff once its availability condition becomes true. This avoids
    permanently losing recent results from the rolling state.
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
    cols += [c for c in ("home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target", "home_corners", "away_corners", "home_fouls", "away_fouls", "home_yellow_cards", "away_yellow_cards", "home_red_cards", "away_red_cards") if c in h.columns]
    if "source_available_at_utc" in h.columns:
        cols.append("source_available_at_utc")
    h_records = h[cols].to_dict("records")
    m_records = m[[c for c in ["match_id", "competition", "season", "season_start", "kickoff_utc", "home_team", "away_team"] if c in m.columns]].to_dict("records")

    team_games: dict[str, deque] = defaultdict(lambda: deque(maxlen=40))
    team_last: dict[str, pd.Timestamp] = {}
    team_last_available: dict[str, pd.Timestamp] = {}
    h2h: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=10))
    elo = {"global": {}, "competition": {}}
    ptr = 0
    block_ptr = 0
    team_max_prior_available: dict[str, pd.Timestamp] = {}
    rows = []
    required_window = max(windows) if windows else 0

    def availability_for(r: dict) -> pd.Timestamp:
        event = r["kickoff_utc"]
        source_available = r.get("source_available_at_utc")
        return source_available if pd.notna(source_available) else result_feature_available_at(event)

    def ingest_until(cutoff: pd.Timestamp) -> None:
        nonlocal ptr
        while ptr < len(h_records):
            r = h_records[ptr]
            event = r["kickoff_utc"]
            if event >= cutoff:
                break
            available = availability_for(r)
            # Do not advance past an unavailable historical event. Because records
            # are chronological, waiting here is conservative and guarantees that
            # the same event can be ingested on a later prediction cutoff.
            if pd.isna(available) or available > cutoff:
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
                entry = {"time": event, "gf": gf, "ga": ga, "points": pts, "venue": venue, "gd": gd, "competition": str(r["competition"]), "available": available}
                for k in STAT_KEYS:
                    entry[k] = _stat_value(r, team, k)
                team_games[team].append(entry)
                team_last[team] = event
                team_last_available[team] = available
            h2h[(home, away)].append(result)
            h2h[(away, home)].append(2 - result if result != 1 else 1)
            ptr += 1

    def update_pit_blockers(cutoff: pd.Timestamp) -> None:
        nonlocal block_ptr
        while block_ptr < len(h_records):
            r = h_records[block_ptr]
            event = r["kickoff_utc"]
            if event >= cutoff:
                break
            available = availability_for(r)
            if pd.notna(available):
                for team in (str(r["home_team"]), str(r["away_team"])):
                    prev = team_max_prior_available.get(team)
                    if prev is None or available > prev:
                        team_max_prior_available[team] = available
            block_ptr += 1

    for r in m_records:
        kickoff = r["kickoff_utc"]
        cutoff = kickoff - pd.Timedelta(minutes=60)
        ingest_until(cutoff)
        update_pit_blockers(cutoff)
        home, away, comp = str(r["home_team"]), str(r["away_team"]), str(r["competition"])
        he = float(elo["global"].get(home, 1500.0)); ae = float(elo["global"].get(away, 1500.0))
        ce = elo["competition"].get(comp, {}); hce = float(ce.get(home, 1500.0)); cae = float(ce.get(away, 1500.0))
        blocker_times = [team_max_prior_available.get(t) for t in (home, away) if t in team_max_prior_available]
        pit_blocked = bool(blocker_times and max(blocker_times) > cutoff)
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
            if pit_blocked:
                hs = {k: np.nan for k in hs}; aws = {k: np.nan for k in aws}
            for k, v in hs.items(): row[f"home_{k}_{w}"] = v
            for k, v in aws.items(): row[f"away_{k}_{w}"] = v
            row[f"gf_diff_{w}"] = hs["gf"] - aws["gf"]
            row[f"ga_diff_{w}"] = hs["ga"] - aws["ga"]
            row[f"points_diff_{w}"] = hs["points"] - aws["points"]
            row[f"win_rate_diff_{w}"] = hs["win_rate"] - aws["win_rate"]
            for k in STAT_KEYS:
                row[f"{k}_diff_{w}"] = hs[f"{k}_avg"] - aws[f"{k}_avg"]
        meetings = [] if pit_blocked else list(h2h[(home, away)])[-5:]
        row["h2h_games_5"] = float(len(meetings))
        row["h2h_home_win_rate_5"] = float(np.mean([x == 0 for x in meetings])) if meetings else np.nan
        row["h2h_draw_rate_5"] = float(np.mean([x == 1 for x in meetings])) if meetings else np.nan
        row["h2h_away_win_rate_5"] = float(np.mean([x == 2 for x in meetings])) if meetings else np.nan
        row["h2h_points_edge_5"] = float(np.mean([3 if x == 0 else 1 if x == 1 else 0 for x in meetings]) - np.mean([3 if x == 2 else 1 if x == 1 else 0 for x in meetings])) if meetings else np.nan
        available_times = [team_last_available[t] for t in (home, away) if t in team_last_available]
        row["feature_source_max_available_at_utc"] = max(available_times) if available_times else pd.NaT
        home_history = list(team_games[home])[-required_window:] if required_window else []
        away_history = list(team_games[away])[-required_window:] if required_window else []
        history_complete = (required_window == 0) or (len(home_history) >= required_window and len(away_history) >= required_window)
        history_available = history_complete and all(x["available"] <= cutoff for x in home_history + away_history)
        row["pit_verified"] = bool(history_available and not pit_blocked)
        rows.append(row)
    return pd.DataFrame(rows)


def add_target(features: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    actual = matches[["match_id", "home_goals", "away_goals"]].copy()
    out = features.merge(actual, on="match_id", how="left", validate="one_to_one")
    out["target"] = np.where(out.home_goals > out.away_goals, 0, np.where(out.home_goals == out.away_goals, 1, 2))
    return out
