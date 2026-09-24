import json
        if pd.isna(row.goals_home) or pd.isna(row.goals_away):
            continue
        # Resolve its conservative post-match availability timestamp.
        match_known = row.known_at
        event = row.date_utc
        team_goals[float(row.home_team_id)].append((event, match_known, float(row.goals_home), float(row.goals_away)))
        team_goals[float(row.away_team_id)].append((event, match_known, float(row.goals_away), float(row.goals_home)))

    p = p.sort_values(["team_id", "player_id", "match_kickoff_utc", "fixture_id"], kind="mergesort")
    rows: list[dict[str, Any]] = []
    diagnostics = {
        "played_fixtures": int(len(played)),
        "player_rows_with_known_at": int(len(p)),
        "team_ids_with_history": int(len(team_goals)),
        "fixtures_with_prior_team_history": 0,
        "candidate_players_before_minutes": 0,
        "candidate_players_with_min_history": 0,
        "candidate_players_with_positive_minutes": 0,
    }
    for fixture in played.itertuples(index=False):
        target_id = str(int(fixture.id))
        target_kickoff = fixture.date_utc
        sides = {
            float(fixture.home_team_id): float(fixture.away_team_id),
            float(fixture.away_team_id): float(fixture.home_team_id),
        }
        for team_id, opponent_id in sides.items():
            prior_team_matches = [
                x for x in team_goals.get(team_id, [])
                if x[0] < target_kickoff and x[1] < target_kickoff
            ]
            prior_opponent_matches = [
                x for x in team_goals.get(opponent_id, [])
                if x[0] < target_kickoff and x[1] < target_kickoff
            ]
            if not prior_team_matches:
                continue
            diagnostics["fixtures_with_prior_team_history"] += 1
            team_slice = prior_team_matches[-lookback_appearances:]
            opponent_slice = prior_opponent_matches[-lookback_appearances:]

            candidate_players = p.loc[
                (p["team_id"] == team_id)
                & (p["match_kickoff_utc"] < target_kickoff)
                & (p["known_at"] < target_kickoff)
            ].copy()
            diagnostics["candidate_players_before_minutes"] += int(candidate_players["player_id"].nunique())
            for player_id, g in candidate_players.groupby("player_id", sort=False):
                g = g.sort_values(["match_kickoff_utc", "fixture_id"], kind="mergesort").tail(lookback_appearances)
                if len(g) >= min_history_appearances:
                    diagnostics["candidate_players_with_min_history"] += 1
                g = g.loc[g["minutes"] > 0].copy()
                if len(g) < min_history_appearances:
                    continue
                diagnostics["candidate_players_with_positive_minutes"] += 1

                age = np.arange(len(g) - 1, -1, -1, dtype=float)
                half = max(float(half_life_appearances), 1.0)
                weights = np.exp(-np.log(2.0) * age / half)
                weights /= max(float(weights.sum()), 1e-12)

                minutes = _numeric(g, "minutes").to_numpy(float)
                rating = _numeric(g, "rating").to_numpy(float)
                goals = _numeric(g, "goals_total").to_numpy(float)
                assists = _numeric(g, "goals_assists").to_numpy(float)
                shots = _numeric(g, "shots_total").to_numpy(float)
                key_passes = _numeric(g, "passes_key").to_numpy(float)
                per90_den = np.clip(minutes, 60.0, None) / 90.0

                valid_rating = np.isfinite(rating)
                rating_ewm = float(np.average(rating[valid_rating], weights=weights[valid_rating])) if valid_rating.any() else np.nan
                starts = g["is_starter"].to_numpy(bool)
                rest_days = float((target_kickoff - g["match_kickoff_utc"].iloc[-1]).total_seconds() / 86400.0)

                team_attack = float(np.mean([x[2] for x in team_slice[-5:]]))
                opponent_defense = float(np.mean([x[3] for x in opponent_slice[-5:]])) if opponent_slice else np.nan
                used_known_at = [g["known_at"].max(), *[x[1] for x in team_slice[-5:]], *[x[1] for x in opponent_slice[-5:]]]
                used_known_at = [ts for ts in used_known_at if pd.notna(ts)]
                if not used_known_at:
                    continue
                feature_available_at = max(used_known_at)
                position = str(g["position"].dropna().iloc[-1]) if g["position"].notna().any() else ""
                rows.append({
                    "match_id": str(target_id),
                    "player_id": str(int(player_id)),
                    "player_name": str(g["player_name"].iloc[-1]),
                    "kickoff_utc": target_kickoff,
                    "feature_available_at_utc": feature_available_at,
                    "pit_verified": bool((g["known_at"] < target_kickoff).all()),
                    "recent_rating_ewm": rating_ewm,
                    "recent_minutes_ewm": float(np.average(minutes, weights=weights)),
                    "recent_goals_per90_ewm": float(np.average(goals / per90_den, weights=weights)),
                    "recent_assists_per90_ewm": float(np.average(assists / per90_den, weights=weights)),
                    "recent_key_passes_per90_ewm": float(np.average(key_passes / per90_den, weights=weights)),
                    "recent_shots_per90_ewm": float(np.average(shots / per90_den, weights=weights)),
                    "recent_starts_rate": float(np.average(starts.astype(float), weights=weights)),
                    "team_attack_strength": team_attack,
                    "opponent_defense_strength": opponent_defense,
                    "position_attack_weight": _position_weight(position),
                    "days_rest": rest_days,
                    "candidate_history_matches": int(g["fixture_id"].nunique()),
                })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(
            "No PIT-safe MOM candidate rows could be constructed; "
            + json.dumps(diagnostics, sort_keys=True)
        )

    # Missing performance statistics remain explicit NaN values. The downstream
    # MOM model learns training-slice-only median imputation plus missingness
    # indicators; no missing statistic is converted to zero here.

    required = {
        "match_id", "player_id", "kickoff_utc", "feature_available_at_utc",
        "pit_verified", "candidate_history_matches",
    }
    if not required.issubset(out.columns):
        raise RuntimeError("MOM feature schema construction failed")
    model_features = [
        "recent_rating_ewm", "recent_minutes_ewm", "recent_goals_per90_ewm",
        "recent_assists_per90_ewm", "recent_key_passes_per90_ewm",