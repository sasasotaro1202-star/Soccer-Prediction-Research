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
        actual["source_available_at_utc"] = pd.to_datetime(
            actual["source_available_at_utc"], utc=True, errors="coerce"
        )
    out = features.merge(actual, on="match_id", how="left", validate="one_to_one")
    # Missing outcomes must remain missing. Treating NaN comparisons as an away win
    # would create synthetic labels and contaminate chronological OOS training.
    home_goals = pd.to_numeric(out["home_goals"], errors="coerce")
    away_goals = pd.to_numeric(out["away_goals"], errors="coerce")
    out["target"] = np.select(
        [
            home_goals.isna() | away_goals.isna(),
            home_goals > away_goals,
            home_goals == away_goals,
        ],
        [
            np.nan,
            0,
            1,
        ],
        default=2,
    )
    return out