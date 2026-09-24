"""Build PIT-safe pre-match player features from a versioned public dataset.

The adapter is deliberately source-agnostic at the file boundary. Callers supply
already-read frames and must preserve the source's `known_at` availability field.
This keeps the research layer free from hidden network assumptions.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_FIXTURE_COLUMNS = {"id", "date_utc", "home_team_id", "away_team_id", "goals_home", "goals_away"}
REQUIRED_PLAYER_COLUMNS = {"fixture_id", "team_id", "player_id", "player_name", "is_starter", "position", "minutes", "rating"}
REQUIRED_PLAYER_STATS_COLUMNS = {"fixture_id", "player_id"}
REQUIRED_KNOWN_AT_COLUMNS = {"fixture_id", "known_at"}
# Source-code provenance and the separately published immutable HF data revision.
SOCCER_DATASET_SOURCE_REPOSITORY = "v-eatpizzanot/soccer-dataset"
SOCCER_DATASET_SOURCE_COMMIT = "af71e692edbda9e4697ce1bda2e06b551f3a0052"
SOCCER_DATASET_REPOSITORY = "eatpizzanot/soccer-dataset"
SOCCER_DATASET_COMMIT = "f0cbe86"
SOCCER_DATASET_VERSION = "1.0.0"
SOCCER_DATASET_LICENSE = "CC-BY-4.0"

def soccer_dataset_pinned_urls() -> dict[str, str]:
    base = f"https://huggingface.co/datasets/{SOCCER_DATASET_REPOSITORY}/resolve/{SOCCER_DATASET_COMMIT}"
    return {
        "fixtures": f"{base}/fixtures.parquet",
        "match_stats": f"{base}/match_stats.parquet",
        "fixture_players": f"{base}/fixture_players.parquet",
        "fixture_players_stats_flat": f"{base}/fixture_players_stats_flat.parquet",
    }


STAT_COLUMNS = (
    "rating",
    "minutes",
    "goals_total",
    "goals_assists",
    "shots_total",
    "passes_key",
)


def _require(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _canonical_identity(value: Any) -> str:
    """Normalize numeric-looking IDs so CSV/PQ string coercion cannot break joins."""
    text = str(value).strip()
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return text
    if np.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return text


def _position_weight(value: Any) -> float:
    code = str(value or "").upper().strip()
    if code in {"F", "FW", "ST", "CF", "LW", "RW"}:
        return 1.0
    if code in {"M", "MF", "AM", "CM", "LM", "RM", "DM"}:
        return 0.85
    if code in {"D", "DF", "CB", "LB", "RB", "LWB", "RWB"}:
        return 0.55
    if code in {"G", "GK"}:
        return 0.25
    return np.nan


def _prepare_inputs(
    fixtures: pd.DataFrame,
    player_matches: pd.DataFrame,
    player_stats: pd.DataFrame,
    match_stats: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require(fixtures, REQUIRED_FIXTURE_COLUMNS, "fixtures")
    _require(player_matches, REQUIRED_PLAYER_COLUMNS, "fixture_players")
    _require(player_stats, REQUIRED_PLAYER_STATS_COLUMNS, "fixture_players_stats_flat")
    _require(match_stats, REQUIRED_KNOWN_AT_COLUMNS, "match_stats")

    f = fixtures.copy()
    p = player_matches.copy()
    ps = player_stats.copy()
    s = match_stats.copy()

    f["id"] = pd.to_numeric(f["id"], errors="coerce")
    f["date_utc"] = pd.to_datetime(f["date_utc"], utc=True, errors="coerce")
    f["home_team_id"] = pd.to_numeric(f["home_team_id"], errors="coerce")
    f["away_team_id"] = pd.to_numeric(f["away_team_id"], errors="coerce")
    f["goals_home"] = _numeric(f, "goals_home")
    f["goals_away"] = _numeric(f, "goals_away")
    f["is_played"] = f.get("is_played", f["goals_home"].notna() & f["goals_away"].notna())
    f["is_played"] = f["is_played"].astype(bool)
    f = f.dropna(subset=["id", "date_utc", "home_team_id", "away_team_id"]).copy()

    p["fixture_id"] = pd.to_numeric(p["fixture_id"], errors="coerce")
    p["team_id"] = pd.to_numeric(p["team_id"], errors="coerce")
    p["player_id"] = pd.to_numeric(p["player_id"], errors="coerce")
    p["player_name"] = p["player_name"].astype("string").str.strip()
    p = p.dropna(subset=["fixture_id", "team_id", "player_id"]).copy()
    for col in ("minutes", "rating"):
        p[col] = _numeric(p, col)

    ps["fixture_id"] = pd.to_numeric(ps["fixture_id"], errors="coerce")
    ps["player_id"] = pd.to_numeric(ps["player_id"], errors="coerce")
    ps = ps.dropna(subset=["fixture_id", "player_id"]).copy()
    if ps.duplicated(["fixture_id", "player_id"]).any():
        raise RuntimeError("fixture_players_stats_flat contains duplicate fixture/player rows")
    flat_map = {
        "games_minutes": "_flat_minutes",
        "games_rating": "_flat_rating",
        "goals_total": "_flat_goals_total",
        "goals_assists": "_flat_goals_assists",
        "shots_total": "_flat_shots_total",
        "passes_key": "_flat_passes_key",
    }
    for src, dst in flat_map.items():
        ps[dst] = _numeric(ps, src)
    p = p.merge(
        ps[["fixture_id", "player_id", *flat_map.values()]],
        on=["fixture_id", "player_id"],
        how="left",
        validate="one_to_one",
    )
    p["minutes"] = p["minutes"].combine_first(p["_flat_minutes"])
    p["rating"] = p["rating"].combine_first(p["_flat_rating"])
    for column in ("goals_total", "goals_assists", "shots_total", "passes_key"):
        p[column] = _numeric(p, column).combine_first(
            _numeric(p, f"_flat_{column}")
        )
    p["position"] = p["position"].astype("string")
    p["is_starter"] = p["is_starter"].astype("boolean")

    s["fixture_id"] = pd.to_numeric(s["fixture_id"], errors="coerce")
    s["known_at"] = pd.to_datetime(s["known_at"], utc=True, errors="coerce")
    s = s.dropna(subset=["fixture_id", "known_at"]).copy()
    if s["fixture_id"].duplicated().any():
        # Multiple provider rows are allowed only when they share the same
        # conservative availability timestamp; otherwise the source is ambiguous.
        dup = s.groupby("fixture_id")["known_at"].nunique(dropna=True)
        if bool((dup > 1).any()):
            raise RuntimeError("match_stats has multiple known_at timestamps for one fixture")

    known = s[["fixture_id", "known_at"]].drop_duplicates("fixture_id")
    p = p.merge(known, on="fixture_id", how="left", validate="many_to_one")
    p = p.merge(
        f[["id", "date_utc", "goals_home", "goals_away", "home_team_id", "away_team_id"]]
        .rename(columns={"id": "fixture_id", "date_utc": "match_kickoff_utc"}),
        on="fixture_id",
        how="left",
        validate="many_to_one",
    )
    return f, p, s


def build_mom_feature_rows(
    fixtures: pd.DataFrame,
    player_matches: pd.DataFrame,
    player_stats: pd.DataFrame,
    match_stats: pd.DataFrame,
    *,
    min_history_appearances: int = 3,
    lookback_appearances: int = 10,
    half_life_appearances: float = 5.0,
) -> pd.DataFrame:
    """Create one pre-match feature row per eligible player-candidate.

    Candidate eligibility is based only on a player's earlier appearances for the
    same team. No target-match lineup, rating, minutes or statistics are used.
    Prior player-match facts must have `known_at < target kickoff`.
    """
    if min_history_appearances < 1 or lookback_appearances < min_history_appearances:
        raise ValueError("invalid player history window")
    f, p, s = _prepare_inputs(fixtures, player_matches, player_stats, match_stats)
    fixture_known = s[["fixture_id", "known_at"]].drop_duplicates("fixture_id")
    played = f.loc[f["is_played"]].merge(
        fixture_known.rename(columns={"fixture_id": "id"}), on="id", how="left", validate="one_to_one"
    ).sort_values("date_utc", kind="mergesort").copy()
    if played.empty:
        raise RuntimeError("No played fixtures available for MOM feature construction")
    played = played.loc[played["known_at"].notna()].copy()

    p = p.loc[p["match_kickoff_utc"].notna() & p["known_at"].notna()].copy()
    if p.empty:
        raise RuntimeError("No player rows have verified known_at availability")

    # Team match history is built from completed fixtures. A fixture becomes usable
    # only at the source-provided known_at timestamp.
    team_goals: dict[float, list[tuple[pd.Timestamp, pd.Timestamp, float, float]]] = defaultdict(list)
    for row in played.itertuples(index=False):
        if pd.isna(row.goals_home) or pd.isna(row.goals_away):
            continue
        # Resolve its conservative post-match availability timestamp.
        match_known = row.known_at
        event = row.date_utc
        team_goals[float(row.home_team_id)].append((event, match_known, float(row.goals_home), float(row.goals_away)))
        team_goals[float(row.away_team_id)].append((event, match_known, float(row.goals_away), float(row.goals_home)))

    p = p.sort_values(["team_id", "player_id", "match_kickoff_utc", "fixture_id"], kind="mergesort")
    rows: list[dict[str, Any]] = []
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
            team_slice = prior_team_matches[-lookback_appearances:]
            opponent_slice = prior_opponent_matches[-lookback_appearances:]

            candidate_players = p.loc[
                (p["team_id"] == team_id)
                & (p["match_kickoff_utc"] < target_kickoff)
                & (p["known_at"] < target_kickoff)
            ].copy()
            for player_id, g in candidate_players.groupby("player_id", sort=False):
                g = g.sort_values(["match_kickoff_utc", "fixture_id"], kind="mergesort").tail(lookback_appearances)
                g = g.loc[g["minutes"] > 0].copy()
                if len(g) < min_history_appearances:
                    continue

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
        raise RuntimeError("No PIT-safe MOM candidate rows could be constructed")

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
        "recent_shots_per90_ewm", "recent_starts_rate", "team_attack_strength",
        "opponent_defense_strength", "position_attack_weight", "days_rest",
    ]
    numeric = out[model_features].apply(pd.to_numeric, errors="coerce")
    # Numeric conversion validates schema, but missing values are deliberately
    # retained for model-side missingness-aware imputation.
    if bool(np.isinf(numeric.to_numpy(dtype=float)).any()):
        raise RuntimeError("MOM feature construction produced infinite values")
    out = out.loc[
        out["pit_verified"]
        & (out["feature_available_at_utc"] < out["kickoff_utc"])
    ].copy()
    return out.reset_index(drop=True)


def attach_mom_labels(
    feature_rows: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    match_id_col: str = "match_id",
    player_id_col: str = "player_id",
) -> pd.DataFrame:
    """Attach an outcome-only MOM label set without treating missing as negative."""
    required = {match_id_col, player_id_col}
    _require(labels, required, "MOM labels")
    x = feature_rows.copy()
    y = labels[[match_id_col, player_id_col]].copy()
    y[match_id_col] = y[match_id_col].map(_canonical_identity).astype("string")
    y[player_id_col] = y[player_id_col].map(_canonical_identity).astype("string")
    y = y.drop_duplicates()
    if y.duplicated(match_id_col).any():
        counts = y.groupby(match_id_col)[player_id_col].nunique()
        if bool((counts > 1).any()):
            raise ValueError("MOM labels contain multiple winners for one match")
    x[match_id_col] = x[match_id_col].map(_canonical_identity).astype("string")
    x[player_id_col] = x[player_id_col].map(_canonical_identity).astype("string")
    x["is_motm"] = 0
    winners = set(zip(y[match_id_col].astype(str), y[player_id_col].astype(str)))
    x["is_motm"] = [
        int((str(m), str(p)) in winners)
        for m, p in zip(x[match_id_col], x[player_id_col])
    ]
    # Every evaluated match must have exactly one winner after candidate filtering.
    counts = x.groupby(match_id_col, sort=False)["is_motm"].sum()
    valid_matches = counts.index[counts == 1]
    return x.loc[x[match_id_col].isin(valid_matches)].reset_index(drop=True)


def mom_data_contract_report(feature_rows: pd.DataFrame) -> dict[str, Any]:
    """Return a machine-readable readiness report; never turns missing data into zero."""
    if feature_rows.empty:
        return {"status": "DEFERRED_NO_PIT_PLAYER_DATA", "matches": 0, "rows": 0}
    pit = feature_rows.get("pit_verified", pd.Series(False, index=feature_rows.index)).astype(bool)
    match_counts = feature_rows.groupby("match_id")["player_id"].nunique()
    return {
        "status": "READY_FOR_OOS" if bool(pit.all()) and bool((match_counts >= 4).all()) else "DEFERRED_INSUFFICIENT_PIT_CANDIDATES",
        "matches": int(feature_rows["match_id"].nunique()),
        "rows": int(len(feature_rows)),
        "pit_verified_rows": int(pit.sum()),
        "matches_with_at_least_4_candidates": int((match_counts >= 4).sum()),
        "minimum_candidates": int(match_counts.min()),
        "fail_closed": True,
        "source_repository": SOCCER_DATASET_REPOSITORY,
        "source_commit": SOCCER_DATASET_COMMIT,
        "source_version": SOCCER_DATASET_VERSION,
        "source_license": SOCCER_DATASET_LICENSE,
    }
