from collections import defaultdict, deque

import numpy as np
import pandas as pd

from src.data.pit_policy import is_available_by_cutoff

ELO_K = 20.0
ELO_HOME_ADV = 55.0
DYNAMIC_ELO_K = 28.0
DYNAMIC_ELO_MEAN_REVERSION_30D = 0.08
NEUTRAL_VENUE_REQUIRED_COMPETITIONS = {"AG_M", "AG_W"}
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
                "gf_ewma": np.nan, "ga_ewma": np.nan, "gd_ewma": np.nan,
                "points_ewma": np.nan, "gd_std": np.nan, "home_rate": np.nan,
                "goal_total_avg": np.nan, "clean_sheet_rate": np.nan,
                "failed_to_score_rate": np.nan}
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
        "gf_ewma": _ewma(gf), "ga_ewma": _ewma(ga), "gd_ewma": _ewma(gd),
        "points_ewma": _ewma(pts),
        "gd_std": float(np.std(gd)) if len(gd) > 1 else 0.0,
        "home_rate": float(np.mean([x["venue"] == "H" for x in recent])),
        "goal_total_avg": float(np.mean([g + a for g, a in zip(gf, ga)])),
        "clean_sheet_rate": float(np.mean([g == 0 for g in ga])),
        "failed_to_score_rate": float(np.mean([g == 0 for g in gf])),
    }
    for k in STAT_KEYS:
        vals = [x.get(k, np.nan) for x in recent]
        out[f"{k}_avg"] = float(np.nanmean(vals)) if any(pd.notna(v) for v in vals) else np.nan
        out[f"{k}_ewma"] = _ewma(vals)
    return out


def _update_elo(elo: dict, home: str, away: str, result: int, competition: str, neutral_venue: bool = False) -> None:
    actual = 1.0 if result == 0 else 0.5 if result == 1 else 0.0
    he = float(elo["global"].get(home, 1500.0)); ae = float(elo["global"].get(away, 1500.0))
    home_adv = 0.0 if neutral_venue else ELO_HOME_ADV
    expected = 1.0 / (1.0 + 10.0 ** (-(he + home_adv - ae) / 400.0))
    delta = ELO_K * (actual - expected)
    elo["global"][home] = he + delta; elo["global"][away] = ae - delta
    ce = elo["competition"].setdefault(competition, {})
    che = float(ce.get(home, 1500.0)); cae = float(ce.get(away, 1500.0))
    expected_c = 1.0 / (1.0 + 10.0 ** (-(che + home_adv - cae) / 400.0))
    delta_c = ELO_K * (actual - expected_c)
    ce[home] = che + delta_c; ce[away] = cae - delta_c


def build_match_features(history: pd.DataFrame, matches: pd.DataFrame, windows=(3, 5, 10, 20)) -> pd.DataFrame:
    """Build chronological PIT-safe features without inventing publication times."""
    h = history.copy()
    h["kickoff_utc"] = pd.to_datetime(h["kickoff_utc"], utc=True, errors="coerce")
    if "source_available_at_utc" in h.columns:
        h["source_available_at_utc"] = pd.to_datetime(h["source_available_at_utc"], utc=True, errors="coerce")
    else:
        h["source_available_at_utc"] = pd.NaT
    if "neutral_venue" not in h.columns:
        h["neutral_venue"] = pd.NA
    h = h.dropna(subset=["kickoff_utc"]).sort_values(
        ["kickoff_utc", "competition", "home_team", "away_team", "match_id"], kind="mergesort"
    ).reset_index(drop=True)
    m = matches.copy()
    neutral_field_present = "neutral_venue" in m.columns
    m["kickoff_utc"] = pd.to_datetime(m["kickoff_utc"], utc=True, errors="coerce")
    m = m.dropna(subset=["kickoff_utc"]).sort_values(
        ["kickoff_utc", "competition", "home_team", "away_team", "match_id"], kind="mergesort"
    ).reset_index(drop=True)

    cols = ["kickoff_utc", "home_team", "away_team", "home_goals", "away_goals", "competition", "match_id", "source_available_at_utc", "neutral_venue"]
    cols += [c for c in ("home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target", "home_corners", "away_corners", "home_fouls", "away_fouls", "home_yellow_cards", "away_yellow_cards", "home_red_cards", "away_red_cards") if c in h.columns]
    h_records = h[cols].to_dict("records")
    m_records = m[[c for c in ["match_id", "competition", "season", "season_start", "kickoff_utc", "home_team", "away_team", "neutral_venue"] if c in m.columns]].to_dict("records")

    team_games: dict[str, deque] = defaultdict(lambda: deque(maxlen=40))
    team_last: dict[str, pd.Timestamp] = {}
    team_last_available: dict[str, pd.Timestamp] = {}
    h2h: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=10))
    elo = {"global": {}, "competition": {}}
    dynamic_elo: dict[str, float] = {}
    dynamic_elo_last: dict[str, pd.Timestamp] = {}
    eligible_indices: set[int] = set()
    availability_order = sorted(
        [i for i, r in enumerate(h_records) if pd.notna(r["source_available_at_utc"])],
        key=lambda i: (h_records[i]["source_available_at_utc"], h_records[i]["kickoff_utc"], str(h_records[i]["match_id"]))
    )
    avail_ptr = 0
    processed_event_max = pd.NaT
    rows = []
    required_window = min(windows) if windows else 0

    def reset_state() -> None:
        nonlocal team_games, team_last, team_last_available, h2h, elo, dynamic_elo, dynamic_elo_last, processed_event_max
        team_games = defaultdict(lambda: deque(maxlen=40))
        team_last = {}
        team_last_available = {}
        h2h = defaultdict(lambda: deque(maxlen=10))
        elo = {"global": {}, "competition": {}}
        dynamic_elo = {}
        dynamic_elo_last = {}
        processed_event_max = pd.NaT

    def apply_row(r: dict) -> None:
        nonlocal processed_event_max
        hg, ag = r["home_goals"], r["away_goals"]
        if pd.isna(hg) or pd.isna(ag):
            return
        event = r["kickoff_utc"]
        available = r["source_available_at_utc"]
        home, away = str(r["home_team"]), str(r["away_team"])
        result = 0 if hg > ag else 1 if hg == ag else 2
        neutral = bool(r.get("neutral_venue", False)) if pd.notna(r.get("neutral_venue", False)) else False
        for team in (home, away):
            current = float(dynamic_elo.get(team, 1500.0))
            last = dynamic_elo_last.get(team)
            if last is not None:
                days = max(float((event - last).total_seconds() / 86400.0), 0.0)
                decay = 1.0 - (1.0 - DYNAMIC_ELO_MEAN_REVERSION_30D) ** (days / 30.0)
                current = 1500.0 + (current - 1500.0) * max(0.0, 1.0 - decay)
            dynamic_elo[team] = current
            dynamic_elo_last[team] = event
        deh = float(dynamic_elo.get(home, 1500.0)); dea = float(dynamic_elo.get(away, 1500.0))
        dynamic_home_adv = 0.0 if neutral else ELO_HOME_ADV
        dynamic_expected = 1.0 / (1.0 + 10.0 ** (-((deh + dynamic_home_adv) - dea) / 400.0))
        actual = 1.0 if result == 0 else 0.5 if result == 1 else 0.0
        delta = DYNAMIC_ELO_K * (actual - dynamic_expected)
        dynamic_elo[home] = deh + delta
        dynamic_elo[away] = dea - delta
        _update_elo(elo, home, away, result, str(r["competition"]), neutral_venue=neutral)
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
        processed_event_max = event if pd.isna(processed_event_max) else max(processed_event_max, event)

    def replay(cutoff: pd.Timestamp) -> None:
        reset_state()
        for idx in sorted(eligible_indices, key=lambda i: (h_records[i]["kickoff_utc"], str(h_records[i]["competition"]), str(h_records[i]["home_team"]), str(h_records[i]["away_team"]), str(h_records[i]["match_id"]))):
            r = h_records[idx]
            if r["kickoff_utc"] < cutoff and pd.notna(r["source_available_at_utc"]) and r["source_available_at_utc"] <= cutoff:
                apply_row(r)

    def prior_window_pit_ok(team: str, cutoff: pd.Timestamp) -> bool:
        if not required_window:
            return True
        history = list(team_games.get(team, ()))
        if len(history) < required_window:
            return False
        recent = history[-required_window:]
        return all(
            pd.notna(x.get("available")) and x["available"] <= cutoff
            and pd.notna(x.get("time")) and x["time"] < cutoff
            for x in recent
        )

    for r in m_records:
        kickoff = r["kickoff_utc"]
        cutoff = kickoff - pd.Timedelta(minutes=60)
        newly_eligible = []
        while avail_ptr < len(availability_order):
            idx = availability_order[avail_ptr]
            available = h_records[idx]["source_available_at_utc"]
            if available > cutoff:
                break
            eligible_indices.add(idx)
            newly_eligible.append(idx)
            avail_ptr += 1
        needs_replay = any(pd.notna(processed_event_max) and h_records[i]["kickoff_utc"] < processed_event_max for i in newly_eligible)
        if newly_eligible and needs_replay:
            replay(cutoff)
        else:
            for idx in sorted(newly_eligible, key=lambda i: (h_records[i]["kickoff_utc"], str(h_records[i]["competition"]), str(h_records[i]["home_team"]), str(h_records[i]["away_team"]), str(h_records[i]["match_id"]))):
                rr = h_records[idx]
                if rr["kickoff_utc"] < cutoff:
                    apply_row(rr)

        home, away, comp = str(r["home_team"]), str(r["away_team"]), str(r["competition"])
        neutral_value = r.get("neutral_venue", pd.NA) if neutral_field_present else pd.NA
        if neutral_field_present:
            neutral_known = pd.notna(neutral_value)
        else:
            neutral_known = comp not in NEUTRAL_VENUE_REQUIRED_COMPETITIONS
        neutral = bool(neutral_value) if neutral_known and pd.notna(neutral_value) else False
        for team in (home, away):
            if team in dynamic_elo:
                last = dynamic_elo_last.get(team)
                if last is not None:
                    days = max(float((cutoff - last).total_seconds() / 86400.0), 0.0)
                    decay = 1.0 - (1.0 - DYNAMIC_ELO_MEAN_REVERSION_30D) ** (days / 30.0)
                    dynamic_elo[team] = 1500.0 + (float(dynamic_elo[team]) - 1500.0) * max(0.0, 1.0 - decay)
                    dynamic_elo_last[team] = cutoff
        deh = float(dynamic_elo.get(home, 1500.0)); dea = float(dynamic_elo.get(away, 1500.0))
        de_home_adv = (0.0 if neutral else ELO_HOME_ADV) if neutral_known else np.nan
        dynamic_home_expected = (1.0 / (1.0 + 10.0 ** (-((deh + de_home_adv) - dea) / 400.0))) if neutral_known else np.nan
        he = float(elo["global"].get(home, 1500.0)); ae = float(elo["global"].get(home, 1500.0)) if False else float(elo["global"].get(away, 1500.0))
        ce = elo["competition"].get(comp, {}); hce = float(ce.get(home, 1500.0)); cae = float(ce.get(away, 1500.0))
        home_advantage = (0.0 if neutral else 1.0) if neutral_known else np.nan
        elo_home_adv = (0.0 if neutral else ELO_HOME_ADV) if neutral_known else np.nan
        row = {"match_id": r["match_id"], "competition": comp, "season": r.get("season"), "season_start": r.get("season_start", np.nan), "kickoff_utc": kickoff, "home_team": home, "away_team": away, "prediction_cutoff_at_utc": cutoff, "neutral_venue": neutral, "neutral_venue_known": neutral_known, "home_advantage": home_advantage, "home_elo": he, "away_elo": ae, "elo_diff": he - ae, "home_comp_elo": hce, "away_comp_elo": cae, "comp_elo_diff": hce - cae, "home_elo_expected": (1.0 / (1.0 + 10.0 ** (-((he + elo_home_adv) - ae) / 400.0))) if neutral_known else np.nan, "home_rest_hours": (cutoff - team_last[home]).total_seconds() / 3600.0 if home in team_last else np.nan, "away_rest_hours": (cutoff - team_last[away]).total_seconds() / 3600.0 if away in team_last else np.nan, "home_dynamic_elo": deh, "away_dynamic_elo": dea, "dynamic_elo_diff": deh - dea, "dynamic_home_elo_expected": dynamic_home_expected}
        row["rest_diff_hours"] = row["home_rest_hours"] - row["away_rest_hours"] if pd.notna(row["home_rest_hours"]) and pd.notna(row["away_rest_hours"]) else np.nan
        row["elo_gap_abs"] = abs(row["elo_diff"])
        pit_blocked = False
        for w in windows:
            hs = _summarize(team_games[home], w); aws = _summarize(team_games[away], w)
            if pit_blocked:
                hs = {k: np.nan for k in hs}; aws = {k: np.nan for k in aws}
            for k, v in hs.items(): row[f"home_{k}_{w}"] = v
            for k, v in aws.items(): row[f"away_{k}_{w}"] = v
            row[f"gf_diff_{w}"] = hs["gf"] - aws["gf"]; row[f"ga_diff_{w}"] = hs["ga"] - aws["ga"]; row[f"points_diff_{w}"] = hs["points"] - aws["points"]; row[f"win_rate_diff_{w}"] = hs["win_rate"] - aws["win_rate"]
            row[f"goal_total_diff_{w}"] = hs["goal_total_avg"] - aws["goal_total_avg"]
            row[f"clean_sheet_diff_{w}"] = hs["clean_sheet_rate"] - aws["clean_sheet_rate"]
            row[f"failed_to_score_diff_{w}"] = hs["failed_to_score_rate"] - aws["failed_to_score_rate"]
            row[f"gd_ewma_diff_{w}"] = hs["gd_ewma"] - aws["gd_ewma"]
            row[f"points_ewma_diff_{w}"] = hs["points_ewma"] - aws["points_ewma"]
            for k in STAT_KEYS:
                row[f"{k}_diff_{w}"] = hs[f"{k}_avg"] - aws[f"{k}_avg"]
        meetings = list(h2h[(home, away)])[-5:]
        row["h2h_games_5"] = float(len(meetings)); row["h2h_home_win_rate_5"] = float(np.mean([x == 0 for x in meetings])) if meetings else np.nan; row["h2h_draw_rate_5"] = float(np.mean([x == 1 for x in meetings])) if meetings else np.nan; row["h2h_away_win_rate_5"] = float(np.mean([x == 2 for x in meetings])) if meetings else np.nan; row["h2h_points_edge_5"] = float(np.mean([3 if x == 0 else 1 if x == 1 else 0 for x in meetings]) - np.mean([3 if x == 2 else 1 if x == 1 else 0 for x in meetings])) if meetings else np.nan
        # PIT-safe momentum and matchup interactions. These use only feature state
        # already replayed up to the prediction cutoff, never the current/future result.
        for metric in ("points_ewma", "gd_ewma", "gf_ewma", "ga_ewma", "win_rate"):
            h3 = row.get(f"home_{metric}_3", np.nan)
            h10 = row.get(f"home_{metric}_10", np.nan)
            a3 = row.get(f"away_{metric}_3", np.nan)
            a10 = row.get(f"away_{metric}_10", np.nan)
            row[f"home_{metric}_momentum_3v10"] = h3 - h10 if pd.notna(h3) and pd.notna(h10) else np.nan
            row[f"away_{metric}_momentum_3v10"] = a3 - a10 if pd.notna(a3) and pd.notna(a10) else np.nan
            row[f"{metric}_momentum_diff_3v10"] = (
                row[f"home_{metric}_momentum_3v10"] - row[f"away_{metric}_momentum_3v10"]
                if pd.notna(row[f"home_{metric}_momentum_3v10"]) and pd.notna(row[f"away_{metric}_momentum_3v10"])
                else np.nan
            )

        h_attack = row.get("home_gf_ewma_5", np.nan)
        a_defense = row.get("away_ga_ewma_5", np.nan)
        a_attack = row.get("away_gf_ewma_5", np.nan)
        h_defense = row.get("home_ga_ewma_5", np.nan)
        if all(pd.notna(v) for v in (h_attack, a_defense, a_attack, h_defense)):
            row["attack_defense_matchup_diff_5"] = (h_attack - a_defense) - (a_attack - h_defense)
            row["attack_defense_matchup_sum_5"] = (h_attack - a_defense) + (a_attack - h_defense)
        else:
            row["attack_defense_matchup_diff_5"] = np.nan
            row["attack_defense_matchup_sum_5"] = np.nan

        home_draw = row.get("home_draw_rate_10", np.nan)
        away_draw = row.get("away_draw_rate_10", np.nan)
        row["draw_tension_10"] = home_draw * away_draw if pd.notna(home_draw) and pd.notna(away_draw) else np.nan

        if pd.notna(row.get("dynamic_elo_diff", np.nan)) and pd.notna(row.get("rest_diff_hours", np.nan)):
            row["strength_rest_interaction"] = float(
                np.tanh(row["dynamic_elo_diff"] / 200.0) * np.tanh(row["rest_diff_hours"] / 48.0)
            )
        else:
            row["strength_rest_interaction"] = np.nan

        available_times = [team_last_available[t] for t in (home, away) if t in team_last_available]
        row["feature_source_max_available_at_utc"] = max(available_times) if available_times else pd.NaT
        home_history = list(team_games[home])[-required_window:] if required_window else []
        away_history = list(team_games[away])[-required_window:] if required_window else []
        history_complete = (required_window == 0) or (len(home_history) >= required_window and len(away_history) >= required_window)
        history_available = history_complete and all(x["available"] <= cutoff for x in home_history + away_history)
        row["pit_verified"] = bool(neutral_known and history_available and prior_window_pit_ok(home, cutoff) and prior_window_pit_ok(away, cutoff))
        rows.append(row)
    return pd.DataFrame(rows)


def add_target(features: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    cols = ["match_id", "home_goals", "away_goals"]
    if "source_available_at_utc" in matches.columns:
        cols.append("source_available_at_utc")
    actual = matches[cols].copy()
    if "source_available_at_utc" in actual.columns:
        actual["source_available_at_utc"] = pd.to_datetime(actual["source_available_at_utc"], utc=True, errors="coerce")
    out = features.merge(actual, on="match_id", how="left", validate="one_to_one")
    home_goals = pd.to_numeric(out["home_goals"], errors="coerce")
    away_goals = pd.to_numeric(out["away_goals"], errors="coerce")
    out["target"] = np.select(
        [
            home_goals.isna() | away_goals.isna(),
            home_goals > away_goals,
            home_goals == away_goals,
        ],
        [np.nan, 0, 1],
        default=2,
    )
    return out
