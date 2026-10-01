
"""PIT-safe research candidate for four-choice Man of the Match prediction.

The current public Global Football (Soccer) Data Lake exposes player ratings,
minutes and starter flags plus the conservative post-match match_stats.known_at
boundary, but no canonical official MOM target in the project pipeline.
This lane therefore uses the explicitly named proxy target:
highest post-match player rating among participants.

Current-fixture player rows are used only to create the matured proxy label.
Prediction features are built only from player history whose known_at is available
by the target fixture cutoff. The lane is research-only and never mutates production.
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.data.competition_catalog import ACTIVE_SCOPE
from src.data.global_datalake_adapter import LEAGUE_TO_COMPETITION

BASE_URL = "https://huggingface.co/datasets/eatpizzanot/soccer-dataset/resolve/main"
FILES = {
    "fixtures": f"{BASE_URL}/fixtures.parquet",
    "leagues": f"{BASE_URL}/leagues.parquet",
    "match_stats": f"{BASE_URL}/match_stats.parquet",
    "fixture_players": f"{BASE_URL}/fixture_players.parquet",
}

PROXY_LABEL = "highest_postmatch_player_rating_among_participants"
MIN_TRAIN_FIXTURES = 600
OOS_BLOCK_FIXTURES = 300
MIN_OOS_BLOCKS = 3
PLAYER_HISTORY_MAX = 20
CANDIDATES_PER_TEAM = 8
MIN_PLAYER_APPEARANCES = 2
MAX_PLAYER_AGE_DAYS = 365.0
EPS = 1e-9

FEATURES = (
    "apps_5",
    "rating_mean_5",
    "rating_ewm_5",
    "minutes_mean_5",
    "start_rate_5",
    "apps_20",
    "rating_mean_20",
    "rating_ewm_20",
    "minutes_mean_20",
    "start_rate_20",
    "last_rating",
    "days_since_last_app",
)

@dataclass
class PlayerHistory:
    rows: deque

def _softmax(values: Iterable[float], temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(list(values), dtype=float)
    if x.size == 0:
        return x
    t = max(float(temperature), 1e-6)
    x = np.nan_to_num(x, nan=-20.0, neginf=-20.0, posinf=20.0)
    z = x / t
    z -= np.max(z)
    ex = np.exp(np.clip(z, -60.0, 60.0))
    total = float(ex.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(len(x), 1.0 / len(x))
    return ex / total

def _proxy_winner(group: pd.DataFrame) -> tuple[str | None, str]:
    d = group.copy()
    d["rating"] = pd.to_numeric(d["rating"], errors="coerce")
    d["minutes"] = pd.to_numeric(d["minutes"], errors="coerce")
    d["player_id"] = d["player_id"].astype("string").str.strip()
    valid = d[
        d["player_id"].notna()
        & d["player_id"].ne("")
        & d["rating"].notna()
        & np.isfinite(d["rating"])
        & d["minutes"].fillna(0).gt(0)
    ].copy()
    if valid.empty:
        return None, ""
    valid = valid.sort_values(["rating", "player_id"], ascending=[False, True], kind="mergesort")
    return str(valid.iloc[0]["player_id"]), str(valid.iloc[0].get("player_name") or "")

def _ewm(values: list[float], alpha: float = 0.35) -> float:
    clean = [float(v) for v in values if pd.notna(v) and np.isfinite(float(v))]
    if not clean:
        return np.nan
    out = clean[0]
    for value in clean[1:]:
        out = alpha * value + (1.0 - alpha) * out
    return float(out)

def _player_features(state: PlayerHistory | None, *, target_time: pd.Timestamp) -> dict[str, float]:
    if state is None or not state.rows:
        return {name: np.nan for name in FEATURES}
    rows = list(state.rows)
    recent5 = rows[-5:]
    recent20 = rows[-20:]

    def vals(group: list[dict[str, Any]], key: str) -> list[float]:
        out: list[float] = []
        for rec in group:
            value = rec.get(key)
            try:
                if pd.notna(value) and np.isfinite(float(value)):
                    out.append(float(value))
            except (TypeError, ValueError):
                continue
        return out

    ratings5 = vals(recent5, "rating")
    ratings20 = vals(recent20, "rating")
    mins5 = vals(recent5, "minutes")
    mins20 = vals(recent20, "minutes")
    starts5 = vals(recent5, "starter")
    starts20 = vals(recent20, "starter")
    last = rows[-1]
    last_time = pd.Timestamp(last["kickoff_utc"])
    age_days = max(float((target_time - last_time).total_seconds() / 86400.0), 0.0)
    return {
        "apps_5": float(len(recent5)),
        "rating_mean_5": float(np.mean(ratings5)) if ratings5 else np.nan,
        "rating_ewm_5": _ewm(ratings5),
        "minutes_mean_5": float(np.mean(mins5)) if mins5 else np.nan,
        "start_rate_5": float(np.mean(starts5)) if starts5 else np.nan,
        "apps_20": float(len(recent20)),
        "rating_mean_20": float(np.mean(ratings20)) if ratings20 else np.nan,
        "rating_ewm_20": _ewm(ratings20),
        "minutes_mean_20": float(np.mean(mins20)) if mins20 else np.nan,
        "start_rate_20": float(np.mean(starts20)) if starts20 else np.nan,
        "last_rating": float(last.get("rating")) if pd.notna(last.get("rating")) else np.nan,
        "days_since_last_app": age_days,
    }

def _player_priority(features: dict[str, float]) -> float:
    rating = float(features.get("rating_ewm_5", 0.0)) if pd.notna(features.get("rating_ewm_5")) else 0.0
    starts = float(features.get("start_rate_5", 0.0)) if pd.notna(features.get("start_rate_5")) else 0.0
    minutes = float(features.get("minutes_mean_5", 0.0)) if pd.notna(features.get("minutes_mean_5")) else 0.0
    apps = float(features.get("apps_20", 0.0) or 0.0)
    return rating + 0.35 * starts + 0.002 * minutes + 0.005 * min(apps, 20.0)

def _candidate_pool(
    team_id: str,
    team_players: dict[str, set[str]],
    states: dict[str, PlayerHistory],
    *,
    target_time: pd.Timestamp,
) -> list[str]:
    ranked: list[tuple[str, float]] = []
    for player_id in team_players.get(str(team_id), set()):
        state = states.get(str(player_id))
        if state is None or not state.rows:
            continue
        # Latest known team identity prevents stale-transfer candidates.
        if str(state.rows[-1].get("team_id")) != str(team_id):
            continue
        features = _player_features(state, target_time=target_time)
        if float(features.get("apps_20", 0.0) or 0.0) < MIN_PLAYER_APPEARANCES:
            continue
        if pd.notna(features.get("days_since_last_app")) and float(features["days_since_last_app"]) > MAX_PLAYER_AGE_DAYS:
            continue
        ranked.append((str(player_id), _player_priority(features)))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return [player_id for player_id, _ in ranked[:CANDIDATES_PER_TEAM]]

def _model_candidates(random_state: int = 42):
    return {
        "logistic": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, C=1.0, random_state=random_state)),
        ]),
        "hist_gb": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=180,
                learning_rate=0.05,
                max_leaf_nodes=12,
                min_samples_leaf=25,
                l2_regularization=1.5,
                random_state=random_state,
            )),
        ]),
    }

def _fit_binary_model(name: str, frame: pd.DataFrame):
    if frame.empty:
        return None
    y = (
        frame["player_id"].astype(str)
        == frame["proxy_winner_player_id"].astype(str)
    ).astype(int)
    if y.nunique() < 2:
        return None
    model = _model_candidates()[name]
    model.fit(frame[list(FEATURES)], y)
    return model

def _model_logits(model, frame: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(frame[list(FEATURES)]), dtype=float).reshape(-1)
    p = np.asarray(model.predict_proba(frame[list(FEATURES)])[:, 1], dtype=float)
    return np.log(np.clip(p, EPS, 1.0) / np.clip(1.0 - p, EPS, 1.0))

def _fixture_scores(case_frames: list[pd.DataFrame], model_name: str, model, temperature: float = 1.0):
    metrics = {"fixtures": 0, "top1": 0, "top4": 0, "mrr_sum": 0.0, "losses": [], "recall": []}
    rows: list[dict[str, Any]] = []
    for frame in case_frames:
        if frame.empty:
            continue
        logits = _model_logits(model, frame)
        probs = _softmax(logits, temperature=temperature)
        order = np.argsort(-probs, kind="stable")
        player_ids = frame["player_id"].astype(str).to_numpy()
        winner = str(frame.iloc[0]["proxy_winner_player_id"])
        positions = np.flatnonzero(player_ids == winner)
        covered = len(positions) > 0
        rank = int(np.where(order == positions[0])[0][0] + 1) if covered else len(frame) + 1
        p_true = float(probs[positions[0]]) if covered else EPS
        metrics["fixtures"] += 1
        metrics["top1"] += int(covered and rank == 1)
        metrics["top4"] += int(covered and rank <= 4)
        metrics["mrr_sum"] += 1.0 / rank if covered else 0.0
        metrics["losses"].append(-math.log(max(p_true, EPS)))
        metrics["recall"].append(bool(covered))
        for out_rank, idx in enumerate(order[:4], start=1):
            r = frame.iloc[int(idx)]
            rows.append({
                "model": model_name,
                "match_id": str(frame.iloc[0]["match_id"]),
                "competition": str(frame.iloc[0]["competition"]),
                "kickoff_utc": str(frame.iloc[0]["kickoff_utc"]),
                "player_id": str(r["player_id"]),
                "player_name": str(r.get("player_name") or ""),
                "rank": int(out_rank),
                "probability": float(probs[idx]),
                "proxy_winner_player_id": winner,
                "proxy_winner_name": str(frame.iloc[0].get("proxy_winner_name") or ""),
                "proxy_winner_in_candidate_pool": bool(covered),
            })
    n = int(metrics["fixtures"])
    return {
        "fixtures": n,
        "top1_hit_rate": float(metrics["top1"] / n) if n else np.nan,
        "top4_hit_rate": float(metrics["top4"] / n) if n else np.nan,
        "mrr": float(metrics["mrr_sum"] / n) if n else np.nan,
        "logloss": float(np.mean(metrics["losses"])) if metrics["losses"] else np.nan,
        "candidate_recall": float(np.mean(metrics["recall"])) if metrics["recall"] else np.nan,
    }, rows

def _baseline_scores(case_frames: list[pd.DataFrame]):
    metrics = {"fixtures": 0, "top1": 0, "top4": 0, "mrr_sum": 0.0, "losses": [], "recall": []}
    rows: list[dict[str, Any]] = []
    for frame in case_frames:
        if frame.empty:
            continue
        logits = pd.to_numeric(frame["rating_ewm_5"], errors="coerce").to_numpy(dtype=float)
        fallback = pd.to_numeric(frame["rating_mean_20"], errors="coerce").to_numpy(dtype=float)
        logits = np.where(np.isfinite(logits), logits, fallback)
        logits = np.where(np.isfinite(logits), logits, 0.0)
        probs = _softmax(logits)
        order = np.argsort(-probs, kind="stable")
        ids = frame["player_id"].astype(str).to_numpy()
        winner = str(frame.iloc[0]["proxy_winner_player_id"])
        pos = np.flatnonzero(ids == winner)
        covered = len(pos) > 0
        rank = int(np.where(order == pos[0])[0][0] + 1) if covered else len(frame) + 1
        p_true = float(probs[pos[0]]) if covered else EPS
        metrics["fixtures"] += 1
        metrics["top1"] += int(covered and rank == 1)
        metrics["top4"] += int(covered and rank <= 4)
        metrics["mrr_sum"] += 1.0 / rank if covered else 0.0
        metrics["losses"].append(-math.log(max(p_true, EPS)))
        metrics["recall"].append(bool(covered))
        for out_rank, idx in enumerate(order[:4], start=1):
            r = frame.iloc[int(idx)]
            rows.append({
                "model": "baseline_rating_ewm",
                "match_id": str(frame.iloc[0]["match_id"]),
                "competition": str(frame.iloc[0]["competition"]),
                "kickoff_utc": str(frame.iloc[0]["kickoff_utc"]),
                "player_id": str(r["player_id"]),
                "player_name": str(r.get("player_name") or ""),
                "rank": int(out_rank),
                "probability": float(probs[idx]),
                "proxy_winner_player_id": winner,
                "proxy_winner_name": str(frame.iloc[0].get("proxy_winner_name") or ""),
                "proxy_winner_in_candidate_pool": bool(covered),
            })
    n = int(metrics["fixtures"])
    return {
        "fixtures": n,
        "top1_hit_rate": float(metrics["top1"] / n) if n else np.nan,
        "top4_hit_rate": float(metrics["top4"] / n) if n else np.nan,
        "mrr": float(metrics["mrr_sum"] / n) if n else np.nan,
        "logloss": float(np.mean(metrics["losses"])) if metrics["losses"] else np.nan,
        "candidate_recall": float(np.mean(metrics["recall"])) if metrics["recall"] else np.nan,
    }, rows

def _pending_updates(group: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "player_id": str(row["player_id"]),
            "team_id": str(row["team_id"]),
            "kickoff_utc": pd.Timestamp(row["kickoff_utc"]),
            "rating": row["rating"],
            "minutes": row["minutes"],
            "starter": float(bool(row["is_starter"])),
        }
        for _, row in group.iterrows()
        if pd.notna(row["player_id"])
    ]

def _build_cases(raw: pd.DataFrame) -> pd.DataFrame:
    required = {
        "fixture_id", "kickoff_utc", "competition", "team_id", "player_id",
        "player_name", "is_starter", "minutes", "rating", "known_at",
        "home_team_id", "away_team_id",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"MOM proxy input missing columns: {missing}")
    d = raw.copy()
    d["fixture_id"] = pd.to_numeric(d["fixture_id"], errors="coerce")
    d["kickoff_utc"] = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    d["known_at"] = pd.to_datetime(d["known_at"], utc=True, errors="coerce")
    d["team_id"] = pd.to_numeric(d["team_id"], errors="coerce")
    d["home_team_id"] = pd.to_numeric(d["home_team_id"], errors="coerce")
    d["away_team_id"] = pd.to_numeric(d["away_team_id"], errors="coerce")
    d["player_id"] = d["player_id"].astype("string").str.strip()
    d["player_name"] = d["player_name"].astype("string").fillna("")
    d["is_starter"] = d["is_starter"].astype("boolean").fillna(False)
    d["minutes"] = pd.to_numeric(d["minutes"], errors="coerce")
    d["rating"] = pd.to_numeric(d["rating"], errors="coerce")
    d["competition"] = d["competition"].astype("string").str.strip().str.upper()
    d = d.dropna(subset=["fixture_id", "kickoff_utc", "known_at", "team_id", "player_id"])
    d = d[d["competition"].isin(set(map(str, ACTIVE_SCOPE)))].copy()
    d = d.sort_values(["kickoff_utc", "fixture_id", "team_id", "player_id"], kind="mergesort").reset_index(drop=True)

    states: dict[str, PlayerHistory] = {}
    team_players: dict[str, set[str]] = defaultdict(set)
    pending: list[tuple[pd.Timestamp, list[dict[str, Any]]]] = []
    case_rows: list[dict[str, Any]] = []

    grouped = d.groupby(
        ["fixture_id", "kickoff_utc", "competition", "home_team_id", "away_team_id"],
        sort=False,
        dropna=False,
    )
    for (_, kickoff, competition, home_team_id, away_team_id), group in grouped:
        cutoff = pd.Timestamp(kickoff) - pd.Timedelta(minutes=60)
        next_pending: list[tuple[pd.Timestamp, list[dict[str, Any]]]] = []
        for known_at, updates in sorted(pending, key=lambda item: item[0]):
            if known_at <= cutoff:
                for rec in updates:
                    pid = str(rec["player_id"])
                    state = states.setdefault(pid, PlayerHistory(deque(maxlen=PLAYER_HISTORY_MAX)))
                    state.rows.append(rec)
                    team_players[str(rec["team_id"])].add(pid)
            else:
                next_pending.append((known_at, updates))
        pending = next_pending

        candidates = []
        for team_id in (home_team_id, away_team_id):
            candidates.extend(
                _candidate_pool(str(team_id), team_players, states, target_time=cutoff)
            )
        candidates = list(dict.fromkeys(candidates))
        winner, winner_name = _proxy_winner(group)
        if len(candidates) >= 4 and winner is not None:
            for pid in candidates:
                state = states.get(pid)
                feat = _player_features(state, target_time=cutoff)
                name_values = group.loc[
                    group["player_id"].astype(str).eq(pid), "player_name"
                ].dropna()
                player_name = str(name_values.iloc[0]) if not name_values.empty else ""
                case_rows.append({
                    "match_id": str(int(fixture_id)),
                    "competition": str(competition),
                    "kickoff_utc": pd.Timestamp(kickoff).isoformat(),
                    "player_id": str(pid),
                    "player_name": player_name,
                    "proxy_winner_player_id": str(winner),
                    "proxy_winner_name": str(winner_name),
                    **feat,
                })

        known_at = pd.Timestamp(group["known_at"].max())
        if pd.isna(known_at):
            continue
        pending.append((known_at, _pending_updates(group)))

    cases = pd.DataFrame(case_rows)
    if cases.empty:
        raise ValueError("No PIT-safe MOM proxy cases were generated")
    cases["label"] = (
        cases["player_id"].astype(str) == cases["proxy_winner_player_id"].astype(str)
    ).astype(int)
    return cases

def _fixture_case_frames(cases: pd.DataFrame, fixture_ids: list[str]) -> list[pd.DataFrame]:
    wanted = set(map(str, fixture_ids))
    frames = []
    for match_id, frame in cases[cases["match_id"].astype(str).isin(wanted)].groupby("match_id", sort=False):
        frames.append(frame.sort_values("player_id", kind="mergesort").reset_index(drop=True))
    frames.sort(key=lambda frame: pd.Timestamp(frame.iloc[0]["kickoff_utc"]))
    return frames

def _oos_blocks(match_ids: list[str], min_train: int, block_size: int):
    ordered = list(dict.fromkeys(map(str, match_ids)))
    start = int(min_train)
    while start < len(ordered):
        end = min(start + int(block_size), len(ordered))
        yield ordered[:start], ordered[start:end]
        start = end

def run_mom_proxy_research(
    cases: pd.DataFrame,
    *,
    min_train_fixtures: int = MIN_TRAIN_FIXTURES,
    oos_block_fixtures: int = OOS_BLOCK_FIXTURES,
    min_blocks: int = MIN_OOS_BLOCKS,
) -> dict[str, Any]:
    required = {
        "match_id", "kickoff_utc", "competition", "player_id",
        "proxy_winner_player_id", *FEATURES,
    }
    missing = sorted(required - set(cases.columns))
    if missing:
        raise ValueError(f"MOM proxy cases missing columns: {missing}")
    d = cases.copy()
    d["kickoff_utc"] = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    d = d.dropna(subset=["kickoff_utc"]).sort_values(
        ["kickoff_utc", "match_id", "player_id"], kind="mergesort"
    ).reset_index(drop=True)
    fixture_times = d[["match_id", "kickoff_utc"]].drop_duplicates().sort_values("kickoff_utc", kind="mergesort")
    fixture_order = fixture_times["match_id"].astype(str).tolist()
    blocks = list(_oos_blocks(fixture_order, min_train_fixtures, oos_block_fixtures))
    if len(blocks) < int(min_blocks):
        raise ValueError(f"Insufficient chronological OOS blocks: {len(blocks)}")

    rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    locked_ids = set(range(max(0, len(blocks) - 2), len(blocks)))

    for block_id, (train_ids, oos_ids) in enumerate(blocks):
        train = d[d["match_id"].astype(str).isin(set(train_ids))].copy()
        oos = d[d["match_id"].astype(str).isin(set(oos_ids))].copy()
        val_cut = max(1, int(len(train_ids) * 0.20))
        if len(train_ids) - val_cut < 100 or val_cut < 30:
            raise ValueError(f"Block {block_id}: chronological train/validation split too small")
        fit_ids = train_ids[:-val_cut]
        val_ids = train_ids[-val_cut:]
        fit = train[train["match_id"].astype(str).isin(set(fit_ids))].copy()
        val = train[train["match_id"].astype(str).isin(set(val_ids))].copy()
        val_frames = _fixture_case_frames(val, val_ids)

        validation_scores: dict[str, dict[str, float]] = {}
        fitted_validation: dict[str, Any] = {}
        for name in _model_candidates():
            model = _fit_binary_model(name, fit)
            if model is None:
                continue
            fitted_validation[name] = model
            score, _ = _fixture_scores(val_frames, name, model)
            validation_scores[name] = score

        baseline_val, _ = _baseline_scores(val_frames)
        validation_scores["baseline_rating_ewm"] = baseline_val
        if not validation_scores:
            raise ValueError(f"Block {block_id}: no trainable MOM proxy method")
        selected = min(
            validation_scores,
            key=lambda name: float(validation_scores[name]["logloss"])
            if np.isfinite(validation_scores[name]["logloss"]) else 1e99,
        )

        oos_frames = _fixture_case_frames(oos, oos_ids)
        if selected == "baseline_rating_ewm":
            selected_metrics, selected_rows = _baseline_scores(oos_frames)
        else:
            model = _fit_binary_model(selected, train)
            if model is None:
                raise ValueError(f"Block {block_id}: selected model could not be fitted on chronological train")
            selected_metrics, selected_rows = _fixture_scores(oos_frames, selected, model)
        baseline_oos, _ = _baseline_scores(oos_frames)

        block_rows.append({
            "block": int(block_id),
            "oos_start": str(oos_frames[0].iloc[0]["kickoff_utc"]) if oos_frames else "",
            "oos_end": str(oos_frames[-1].iloc[0]["kickoff_utc"]) if oos_frames else "",
            "selected_model": str(selected),
            "fixtures": int(selected_metrics["fixtures"]),
            "top1_hit_rate": selected_metrics["top1_hit_rate"],
            "top4_hit_rate": selected_metrics["top4_hit_rate"],
            "mrr": selected_metrics["mrr"],
            "logloss": selected_metrics["logloss"],
            "candidate_recall": selected_metrics["candidate_recall"],
            "baseline_top1_hit_rate": baseline_oos["top1_hit_rate"],
            "baseline_top4_hit_rate": baseline_oos["top4_hit_rate"],
            "baseline_mrr": baseline_oos["mrr"],
            "baseline_logloss": baseline_oos["logloss"],
            "baseline_candidate_recall": baseline_oos["candidate_recall"],
            "top4_delta": float(selected_metrics["top4_hit_rate"] - baseline_oos["top4_hit_rate"]),
            "logloss_delta": float(selected_metrics["logloss"] - baseline_oos["logloss"]),
            "locked_block": bool(block_id in locked_ids),
        })
        rows.extend(selected_rows)

    block_df = pd.DataFrame(block_rows)
    dev = block_df[~block_df["locked_block"]]
    locked = block_df[block_df["locked_block"]]
    gates = {
        "development_top4_improvement": bool(
            not dev.empty and dev["top4_hit_rate"].mean() > dev["baseline_top4_hit_rate"].mean()
        ),
        "development_logloss_improvement": bool(
            not dev.empty and dev["logloss"].mean() < dev["baseline_logloss"].mean()
        ),
        "locked_non_regression": bool(
            not locked.empty
            and (locked["top4_hit_rate"] >= locked["baseline_top4_hit_rate"]).all()
            and (locked["logloss"] <= locked["baseline_logloss"]).all()
        ),
        "promotion": False,
    }
    state = {
        "schema_version": 1,
        "status": "READY",
        "proxy_label": PROXY_LABEL,
        "production_usable": False,
        "locked_oos_tuning": False,
        "oos_blocks": int(len(block_df)),
        "case_rows": int(len(cases)),
        "development": {
            "top1_hit_rate": float(dev["top1_hit_rate"].mean()) if not dev.empty else np.nan,
            "baseline_top1_hit_rate": float(dev["baseline_top1_hit_rate"].mean()) if not dev.empty else np.nan,
            "top4_hit_rate": float(dev["top4_hit_rate"].mean()) if not dev.empty else np.nan,
            "baseline_top4_hit_rate": float(dev["baseline_top4_hit_rate"].mean()) if not dev.empty else np.nan,
            "mrr": float(dev["mrr"].mean()) if not dev.empty else np.nan,
            "baseline_mrr": float(dev["baseline_mrr"].mean()) if not dev.empty else np.nan,
            "logloss": float(dev["logloss"].mean()) if not dev.empty else np.nan,
            "baseline_logloss": float(dev["baseline_logloss"].mean()) if not dev.empty else np.nan,
            "candidate_recall": float(dev["candidate_recall"].mean()) if not dev.empty else np.nan,
        },
        "locked": {
            "top1_hit_rate": float(locked["top1_hit_rate"].mean()) if not locked.empty else np.nan,
            "baseline_top1_hit_rate": float(locked["baseline_top1_hit_rate"].mean()) if not locked.empty else np.nan,
            "top4_hit_rate": float(locked["top4_hit_rate"].mean()) if not locked.empty else np.nan,
            "baseline_top4_hit_rate": float(locked["baseline_top4_hit_rate"].mean()) if not locked.empty else np.nan,
            "mrr": float(locked["mrr"].mean()) if not locked.empty else np.nan,
            "baseline_mrr": float(locked["baseline_mrr"].mean()) if not locked.empty else np.nan,
            "logloss": float(locked["logloss"].mean()) if not locked.empty else np.nan,
            "baseline_logloss": float(locked["baseline_logloss"].mean()) if not locked.empty else np.nan,
            "candidate_recall": float(locked["candidate_recall"].mean()) if not locked.empty else np.nan,
        },
        "gates": gates,
        "method_blocks": block_rows,
        "output_contract": {
            "top_k": 4,
            "probabilities_sum_over_full_candidate_pool": True,
            "displayed_top4_probabilities_not_forced_to_sum_to_one": True,
        },
        "teacher_contract": {
            "official_mom_label_available": False,
            "proxy_label": PROXY_LABEL,
            "current_fixture_player_rows_used_for_training_features": False,
        },
    }
    return {"state": state, "rows": rows, "blocks": block_rows}

def download_source(cache_dir: str = "cache/mom_proxy") -> dict[str, Path]:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for name, url in FILES.items():
        path = cache / f"{name}.parquet"
        if path.is_file() and path.stat().st_size > 0:
            out[name] = path
            continue
        last_error: Exception | None = None
        for attempt in range(1, 4):
            tmp = path.with_suffix(path.suffix + ".tmp")
            try:
                import requests
                response = requests.get(
                    url,
                    timeout=120,
                    headers={"User-Agent": "SoccerPredictionResearch/1.0 MOM proxy research"},
                    stream=True,
                )
                response.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
                tmp.replace(path)
                break
            except Exception as exc:
                last_error = exc
                if tmp.exists():
                    tmp.unlink()
                if attempt < 3:
                    time.sleep(2 * attempt)
        if last_error is not None:
            raise RuntimeError(f"failed to download {url}") from last_error
        out[name] = path
    return out

def load_source_cases(*, start_year: int = 2012, end_year: int = 2025, cache_dir: str = "cache/mom_proxy") -> pd.DataFrame:
    paths = download_source(cache_dir)
    fixtures = pd.read_parquet(
        paths["fixtures"],
        columns=["id", "date_utc", "league_id", "home_team_id", "away_team_id"],
        engine="pyarrow",
    )
    leagues = pd.read_parquet(paths["leagues"], columns=["id", "name"], engine="pyarrow")
    stats = pd.read_parquet(paths["match_stats"], columns=["fixture_id", "known_at"], engine="pyarrow")
    players = pd.read_parquet(
        paths["fixture_players"],
        columns=["fixture_id", "team_id", "player_id", "player_name", "is_starter", "minutes", "rating"],
        engine="pyarrow",
    )
    fixtures["date_utc"] = pd.to_datetime(fixtures["date_utc"], utc=True, errors="coerce")
    fixtures["league_id"] = pd.to_numeric(fixtures["league_id"], errors="coerce")
    leagues["competition"] = leagues["name"].map(LEAGUE_TO_COMPETITION)
    leagues = leagues[leagues["competition"].astype(str).isin(set(map(str, ACTIVE_SCOPE)))]
    meta = fixtures.merge(
        leagues[["id", "competition"]].rename(columns={"id": "league_id"}),
        on="league_id",
        how="inner",
        validate="many_to_one",
    )
    meta["home_team_id"] = pd.to_numeric(meta["home_team_id"], errors="coerce")
    meta["away_team_id"] = pd.to_numeric(meta["away_team_id"], errors="coerce")
    meta = meta[
        meta["date_utc"].notna()
        & meta["date_utc"].dt.year.between(int(start_year), int(end_year))
    ].copy()
    stats["fixture_id"] = pd.to_numeric(stats["fixture_id"], errors="coerce")
    stats["known_at"] = pd.to_datetime(stats["known_at"], utc=True, errors="coerce")
    stats = stats.dropna(subset=["fixture_id", "known_at"])
    if stats["fixture_id"].duplicated().any():
        raise RuntimeError("MOM proxy match_stats contains duplicate fixture_id values")
    merged = players.merge(
        meta[["id", "competition", "date_utc", "home_team_id", "away_team_id"]].rename(
            columns={"id": "fixture_id", "date_utc": "kickoff_utc"}
        ),
        on="fixture_id",
        how="inner",
        validate="many_to_one",
    ).merge(
        stats[["fixture_id", "known_at"]],
        on="fixture_id",
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("MOM proxy source produced no active-scope player rows")
    return merged

def save_mom_proxy_artifacts(
    *,
    start_year: int = 2012,
    end_year: int = 2025,
    output_dir: str = "artifacts/mom_proxy",
    cache_dir: str = "cache/mom_proxy",
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = load_source_cases(start_year=start_year, end_year=end_year, cache_dir=cache_dir)
    cases = _build_cases(raw)
    result = run_mom_proxy_research(cases)
    pd.DataFrame(result["blocks"]).to_csv(out / "mom_proxy_oos.csv", index=False)
    pd.DataFrame(result["rows"]).to_csv(out / "mom_proxy_cases.csv", index=False)
    state = dict(result["state"])
    state["source_url"] = "https://huggingface.co/datasets/eatpizzanot/soccer-dataset"
    state["source_license"] = "CC-BY-4.0"
    state["feature_count"] = len(FEATURES)
    (out / "mom_proxy_status.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return state

if __name__ == "__main__":
    print(json.dumps(save_mom_proxy_artifacts(), ensure_ascii=False, indent=2, default=str))
