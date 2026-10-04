"""Research-only dynamic match-state, event-hazard and scenario engine.

The module is intentionally isolated from the production prediction path. It
accepts already-replayed, PIT-verified in-play snapshots and estimates the next
discrete event. Future score/discipline paths are then propagated through a
deterministic probability tree. No retrieved-at timestamp is treated as source
availability evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EVENT_TYPES = (
    "HOME_GOAL",
    "AWAY_GOAL",
    "HOME_RED",
    "AWAY_RED",
    "NO_EVENT",
)
_EVENT_SET = set(EVENT_TYPES)

REQUIRED_COLUMNS = {
    "match_id",
    "kickoff_utc",
    "prediction_cutoff_utc",
    "event_time_utc",
    "source_available_at_utc",
    "pit_verified",
    "home_score",
    "away_score",
    "home_red_cards",
    "away_red_cards",
    "next_event_type",
    "next_event_time_utc",
    "hazard_window_minutes",
    "label_available_at_utc",
    "final_home_goals",
    "final_away_goals",
}

STATIC_OPTIONAL_COLUMNS = (
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_pre_match_prob",
    "draw_pre_match_prob",
    "away_pre_match_prob",
)

BASE_FEATURES = (
    "elapsed_minute",
    "remaining_minutes",
    "score_diff",
    "score_total",
    "home_leading",
    "away_leading",
    "draw_state",
    "home_red_cards",
    "away_red_cards",
    "red_diff",
    "home_red_any",
    "away_red_any",
    "late_game",
    "score_diff_abs",
)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def _utc(series: pd.Series, name: str) -> pd.Series:
    out = pd.to_datetime(series, utc=True, errors="coerce")
    if out.isna().any():
        raise ValueError(f"{name} contains invalid timestamps")
    return out


def validate_snapshot_contract(
    frame: pd.DataFrame,
    *,
    require_future_labels: bool = True,
) -> pd.DataFrame:
    """Fail closed on identity, timing, label and state-integrity violations."""
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"match-state snapshot missing required columns: {missing}")

    d = frame.copy()
    for col in (
        "kickoff_utc",
        "prediction_cutoff_utc",
        "event_time_utc",
        "source_available_at_utc",
        "label_available_at_utc",
    ):
        d[col] = _utc(d[col], col)
    raw_next_time = d["next_event_time_utc"].copy()
    parsed_next_time = pd.to_datetime(raw_next_time, utc=True, errors="coerce")
    raw_text = raw_next_time.astype("string").fillna("").str.strip()
    invalid_next_time = raw_text.ne("") & parsed_next_time.isna()
    if invalid_next_time.any():
        raise ValueError("next_event_time_utc contains invalid timestamps")
    d["next_event_time_utc"] = parsed_next_time

    if d["match_id"].isna().any() or d["match_id"].astype(str).str.strip().eq("").any():
        raise ValueError("match_id contains missing/empty values")
    if d[["match_id", "prediction_cutoff_utc"]].duplicated().any():
        raise ValueError("duplicate match_id/prediction_cutoff_utc snapshots")

    if not d["pit_verified"].map(_parse_bool).all():
        raise ValueError("unverified PIT row present")
    if (d["source_available_at_utc"] > d["prediction_cutoff_utc"]).any():
        raise ValueError("source availability after prediction cutoff")
    if (d["event_time_utc"] > d["prediction_cutoff_utc"]).any():
        raise ValueError("state event time is after prediction cutoff")
    if (d["prediction_cutoff_utc"] < d["kickoff_utc"]).any():
        raise ValueError("pregame snapshots are not supported by this in-play contract")
    if (d["prediction_cutoff_utc"] > d["kickoff_utc"] + pd.Timedelta(minutes=120)).any():
        raise ValueError("prediction cutoff exceeds supported 120-minute post-kickoff window")

    d["hazard_window_minutes"] = pd.to_numeric(d["hazard_window_minutes"], errors="coerce")
    if d["hazard_window_minutes"].isna().any() or not np.isfinite(d["hazard_window_minutes"].to_numpy(float)).all():
        raise ValueError("hazard_window_minutes contains invalid values")
    if ((d["hazard_window_minutes"] <= 0) | (d["hazard_window_minutes"] > 15)).any():
        raise ValueError("hazard_window_minutes must be in (0, 15]")
    if d["hazard_window_minutes"].nunique(dropna=False) != 1:
        raise ValueError("hazard_window_minutes must be constant within one dataset")
    if require_future_labels and (
        d["label_available_at_utc"] <= d["prediction_cutoff_utc"]
    ).any():
        raise ValueError("label is available at/before prediction cutoff")

    event_values = d["next_event_type"].astype(str).str.strip().str.upper()
    if (~event_values.isin(_EVENT_SET)).any():
        bad = sorted(event_values[~event_values.isin(_EVENT_SET)].unique().tolist())
        raise ValueError(f"unsupported next_event_type values: {bad}")
    next_time = d["next_event_time_utc"]
    cutoff = d["prediction_cutoff_utc"]
    window = pd.to_timedelta(d["hazard_window_minutes"], unit="m")
    non_none = event_values != "NO_EVENT"
    if next_time[non_none].isna().any() or (next_time[non_none] <= cutoff[non_none]).any():
        raise ValueError("non-NO_EVENT label must have next_event_time_utc after prediction cutoff")
    if (next_time[non_none] > (cutoff[non_none] + window[non_none])).any():
        raise ValueError("next event falls outside hazard window")
    if next_time[~non_none].notna().any():
        raise ValueError("NO_EVENT label must have missing next_event_time_utc")

    numeric = [
        "home_score",
        "away_score",
        "home_red_cards",
        "away_red_cards",
        "final_home_goals",
        "final_away_goals",
    ]
    for col in numeric:
        d[col] = pd.to_numeric(d[col], errors="coerce")
        if d[col].isna().any() or not np.isfinite(d[col].to_numpy(float)).all():
            raise ValueError(f"{col} contains invalid numeric values")
    if (d[numeric[:4]] < 0).any().any():
        raise ValueError("score/card state cannot be negative")
    if (d["final_home_goals"] < 0).any() or (d["final_away_goals"] < 0).any():
        raise ValueError("final goals cannot be negative")
    if (
        (d["home_score"] > d["final_home_goals"]).any()
        or (d["away_score"] > d["final_away_goals"]).any()
    ):
        raise ValueError("current score exceeds final label")

    d["pit_verified"] = d["pit_verified"].map(_parse_bool)
    d["next_event_type"] = event_values
    return d.sort_values(
        ["prediction_cutoff_utc", "match_id"], kind="mergesort"
    ).reset_index(drop=True)


def build_state_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create only state variables known at the prediction cutoff."""
    d = frame.copy()
    kickoff = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    cutoff = pd.to_datetime(d["prediction_cutoff_utc"], utc=True, errors="coerce")
    if kickoff.isna().any() or cutoff.isna().any():
        raise ValueError("invalid state timing")

    elapsed = (cutoff - kickoff).dt.total_seconds() / 60.0
    elapsed = np.clip(elapsed.to_numpy(float), 0.0, 120.0)
    d["elapsed_minute"] = elapsed
    d["remaining_minutes"] = np.maximum(0.0, 90.0 - elapsed)

    d["score_diff"] = d["home_score"].to_numpy(float) - d["away_score"].to_numpy(float)
    d["score_total"] = d["home_score"].to_numpy(float) + d["away_score"].to_numpy(float)
    d["home_leading"] = (d["score_diff"] > 0).astype(float)
    d["away_leading"] = (d["score_diff"] < 0).astype(float)
    d["draw_state"] = (d["score_diff"] == 0).astype(float)

    d["red_diff"] = (
        d["home_red_cards"].to_numpy(float) - d["away_red_cards"].to_numpy(float)
    )
    d["home_red_any"] = (d["home_red_cards"] > 0).astype(float)
    d["away_red_any"] = (d["away_red_cards"] > 0).astype(float)
    d["late_game"] = (elapsed >= 75.0).astype(float)
    d["score_diff_abs"] = np.abs(d["score_diff"].to_numpy(float))
    return d


def select_model_features(frame: pd.DataFrame) -> list[str]:
    features = list(BASE_FEATURES)
    features.extend(c for c in STATIC_OPTIONAL_COLUMNS if c in frame.columns)
    return features


def fit_hazard_model(
    training: pd.DataFrame,
    *,
    as_of_cutoff: pd.Timestamp,
    method: str = "logistic",
    min_rows: int = 120,
    feature_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit a PIT-safe next-event classifier on already-matured historical rows."""
    d = validate_snapshot_contract(training)
    if not set(BASE_FEATURES).issubset(d.columns):
        d = build_state_features(d)
    as_of = pd.Timestamp(as_of_cutoff)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    else:
        as_of = as_of.tz_convert("UTC")

    d = d[d["label_available_at_utc"] <= as_of].copy()
    d = d[d["prediction_cutoff_utc"] < d["label_available_at_utc"]].copy()
    if len(d) < min_rows:
        raise ValueError(
            f"insufficient matured hazard training rows: {len(d)} < {min_rows}"
        )
    if d["match_id"].nunique() < 20:
        raise ValueError("insufficient distinct historical matches for hazard training")

    features = list(feature_columns or select_model_features(d))
    missing_features = [c for c in features if c not in d.columns]
    if missing_features:
        raise ValueError(f"missing hazard feature columns: {missing_features}")

    x = d[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = d["next_event_type"].astype(str).to_numpy()
    if not np.isfinite(np.nan_to_num(x, nan=0.0)).all():
        raise ValueError("hazard features contain non-finite values")
    if len(set(y.tolist())) < 2:
        raise ValueError("hazard training requires at least two event classes")
    if method not in {"logistic", "histgb"}:
        raise ValueError(f"unsupported hazard method: {method}")
    if method == "histgb" and len(d) < 250:
        raise ValueError("histgb challenger requires at least 250 training rows")

    if method == "logistic":
        estimator = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.5,
                        max_iter=2500,
                        solver="lbfgs",
                        random_state=2401,
                    ),
                ),
            ]
        )
    else:
        estimator = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=180,
                        max_leaf_nodes=15,
                        min_samples_leaf=20,
                        l2_regularization=1.0,
                        random_state=2401,
                    ),
                ),
            ]
        )

    estimator.fit(x, y)
    return {
        "schema_version": 1,
        "model_type": "discrete_time_hazard",
        "method": method,
        "feature_columns": features,
        "event_types": list(EVENT_TYPES),
        "training_rows": int(len(d)),
        "training_matches": int(d["match_id"].nunique()),
        "as_of_cutoff_utc": as_of.isoformat(),
        "hazard_window_minutes": float(d["hazard_window_minutes"].iloc[0]),
        "model": estimator,
        "research_only": True,
    }


def predict_hazard(model: Mapping[str, Any], state: Mapping[str, Any]) -> np.ndarray:
    features = list(model["feature_columns"])
    row = pd.DataFrame([{c: state.get(c, np.nan) for c in features}])
    raw = np.asarray(
        model["model"].predict_proba(row[features].to_numpy(float))[0],
        dtype=float,
    )
    classes = list(model["model"].classes_)
    probs = np.full(len(EVENT_TYPES), 1e-9, dtype=float)
    for idx, cls in enumerate(classes):
        name = str(cls)
        if name in _EVENT_SET:
            probs[EVENT_TYPES.index(name)] = max(raw[idx], 1e-9)
    probs /= probs.sum()
    return probs


def _state_refresh(state: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(state)
    diff = float(out.get("home_score", 0.0)) - float(out.get("away_score", 0.0))
    out["score_diff"] = diff
    out["score_total"] = float(out.get("home_score", 0.0)) + float(
        out.get("away_score", 0.0)
    )
    out["home_leading"] = float(diff > 0)
    out["away_leading"] = float(diff < 0)
    out["draw_state"] = float(diff == 0)
    out["red_diff"] = float(out.get("home_red_cards", 0.0)) - float(
        out.get("away_red_cards", 0.0)
    )
    out["home_red_any"] = float(float(out.get("home_red_cards", 0.0)) > 0)
    out["away_red_any"] = float(float(out.get("away_red_cards", 0.0)) > 0)
    out["score_diff_abs"] = abs(diff)
    out["late_game"] = float(float(out.get("elapsed_minute", 0.0)) >= 75.0)
    out["remaining_minutes"] = max(
        0.0, 90.0 - float(out.get("elapsed_minute", 0.0))
    )
    return out


def _transition(state: Mapping[str, Any], event: str) -> dict[str, Any]:
    out = dict(state)
    if event == "HOME_GOAL":
        out["home_score"] = float(out.get("home_score", 0.0)) + 1.0
    elif event == "AWAY_GOAL":
        out["away_score"] = float(out.get("away_score", 0.0)) + 1.0
    elif event == "HOME_RED":
        out["home_red_cards"] = min(
            float(out.get("home_red_cards", 0.0)) + 1.0, 3.0
        )
    elif event == "AWAY_RED":
        out["away_red_cards"] = min(
            float(out.get("away_red_cards", 0.0)) + 1.0, 3.0
        )
    return _state_refresh(out)


def propagate_scenarios(
    model: Mapping[str, Any],
    initial_state: Mapping[str, Any],
    *,
    horizon_minutes: float | None = None,
    step_minutes: float | None = None,
    max_states: int = 10000,
) -> dict[str, Any]:
    """Propagate a distribution of future score/discipline paths."""
    if step_minutes is None:
        step_minutes = float(model.get("hazard_window_minutes", 5.0))
    if step_minutes <= 0 or step_minutes > 15:
        raise ValueError("step_minutes must be in (0, 15]")
    if max_states < 10:
        raise ValueError("max_states must be at least 10")

    start = _state_refresh(initial_state)
    elapsed = float(start.get("elapsed_minute", 0.0))
    remaining = float(start.get("remaining_minutes", max(0.0, 90.0 - elapsed)))
    horizon = remaining if horizon_minutes is None else min(
        float(horizon_minutes), remaining
    )
    if horizon < 0:
        raise ValueError("horizon_minutes cannot be negative")

    buckets: dict[tuple[int, int, int, int], float] = {
        (
            int(round(float(start.get("home_score", 0.0)))),
            int(round(float(start.get("away_score", 0.0)))),
            int(round(float(start.get("home_red_cards", 0.0)))),
            int(round(float(start.get("away_red_cards", 0.0)))),
        ): 1.0
    }

    pruned_mass = 0.0
    elapsed_cursor = elapsed
    while elapsed_cursor < elapsed + horizon - 1e-9:
        dt = min(step_minutes, elapsed + horizon - elapsed_cursor)
        new_buckets: dict[tuple[int, int, int, int], float] = {}

        for key, mass in buckets.items():
            hs, aws, hr, ar = key
            current = dict(start)
            current.update(
                {
                    "elapsed_minute": elapsed_cursor,
                    "remaining_minutes": max(0.0, 90.0 - elapsed_cursor),
                    "home_score": float(hs),
                    "away_score": float(aws),
                    "home_red_cards": float(hr),
                    "away_red_cards": float(ar),
                }
            )
            current = _state_refresh(current)
            probs = predict_hazard(model, current)

            for idx, event in enumerate(EVENT_TYPES):
                p = float(probs[idx])
                if event == "NO_EVENT":
                    nxt = key
                else:
                    tr = _transition(current, event)
                    nxt = (
                        int(round(tr["home_score"])),
                        int(round(tr["away_score"])),
                        int(round(tr["home_red_cards"])),
                        int(round(tr["away_red_cards"])),
                    )
                new_buckets[nxt] = new_buckets.get(nxt, 0.0) + mass * p

        if len(new_buckets) > max_states:
            ranked = sorted(new_buckets.items(), key=lambda kv: (-kv[1], kv[0]))
            kept = ranked[:max_states]
            discarded = sum(v for _, v in ranked[max_states:])
            pruned_mass += float(discarded)
            buckets = dict(kept)
        else:
            buckets = new_buckets

        elapsed_cursor += dt

    total_mass = float(sum(buckets.values()))
    if total_mass <= 0:
        raise RuntimeError("scenario propagation lost all probability mass")

    normalized = {k: v / total_mass for k, v in buckets.items()}
    home_win = draw = away_win = 0.0
    score_rows: list[tuple[int, int, float]] = []

    for (hs, aws, _, _), p in normalized.items():
        score_rows.append((hs, aws, p))
        if hs > aws:
            home_win += p
        elif hs == aws:
            draw += p
        else:
            away_win += p
    score_rows.sort(key=lambda x: (-x[2], x[0], x[1]))

    return {
        "schema_version": 1,
        "status": "OK",
        "outcome_probabilities": {
            "home": float(home_win),
            "draw": float(draw),
            "away": float(away_win),
        },
        "top_scores": [
            {
                "home_goals": int(hs),
                "away_goals": int(aws),
                "probability": float(p),
            }
            for hs, aws, p in score_rows[:10]
        ],
        "state_count": int(len(normalized)),
        "pruned_mass": float(pruned_mass),
        "residual_mass_before_normalization": float(total_mass),
        "simulation": {
            "method": "deterministic_scenario_propagation",
            "step_minutes": float(step_minutes),
            "elapsed_start_minute": float(start.get("elapsed_minute", 0.0)),
            "horizon_minutes": float(horizon),
        },
    }



def _chronological_match_blocks(frame: pd.DataFrame, *, min_test_matches: int = 10, max_folds: int = 8) -> list[tuple[list[str], list[str]]]:
    """Create expanding match-level folds; snapshots from one match never split across train/test."""
    if min_test_matches < 1 or max_folds < 1:
        raise ValueError("min_test_matches/max_folds must be positive")
    starts = (
        frame.groupby("match_id", sort=False)["prediction_cutoff_utc"]
        .min()
        .sort_values(kind="mergesort")
    )
    match_ids = starts.index.astype(str).tolist()
    if len(match_ids) < (min_test_matches * 2):
        return []
    test_size = max(min_test_matches, len(match_ids) // max_folds)
    folds = []
    cursor = test_size
    while cursor + min_test_matches <= len(match_ids):
        train = match_ids[:cursor]
        test = match_ids[cursor: cursor + test_size]
        if len(test) < min_test_matches:
            break
        folds.append((train, test))
        cursor += test_size
        if len(folds) >= max_folds:
            break
    return folds


def _safe_multiclass_logloss(y: Sequence[str], p: np.ndarray, classes: Sequence[str]) -> float:
    yy = np.asarray(y, dtype=str)
    probs = np.asarray(p, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != len(classes) or len(probs) != len(yy):
        raise ValueError("hazard metric shapes are invalid")
    index = {str(name): idx for idx, name in enumerate(classes)}
    if any(str(v) not in index for v in yy):
        raise ValueError("hazard target contains an unknown class")
    target_idx = np.asarray([index[str(v)] for v in yy], dtype=int)
    selected = np.clip(probs[np.arange(len(yy)), target_idx], 1e-12, 1.0)
    return float(np.mean(-np.log(selected)))


def evaluate_hazard_chronological_oos(
    frame: pd.DataFrame,
    *,
    method: str = "logistic",
    min_training_matches: int = 60,
    min_test_matches: int = 10,
    max_folds: int = 8,
) -> dict[str, Any]:
    """Prequential match-level OOS for next-event hazard.

    Each fold trains only on complete earlier matches. A later snapshot from the
    same match can never enter training for an earlier snapshot test row.
    """
    d = build_state_features(validate_snapshot_contract(frame))
    folds = _chronological_match_blocks(
        d,
        min_test_matches=min_test_matches,
        max_folds=max_folds,
    )
    if not folds:
        return {
            "status": "INSUFFICIENT_MATCH_HISTORY",
            "oos_claimed": False,
            "production_usable": False,
            "folds": [],
        }

    by_match = {
        str(match_id): group.copy()
        for match_id, group in d.groupby("match_id", sort=False)
    }
    fold_metrics = []
    all_y = []
    all_p = []

    for fold_number, (train_ids, test_ids) in enumerate(folds, start=1):
        train = pd.concat([by_match[mid] for mid in train_ids], ignore_index=True)
        test = pd.concat([by_match[mid] for mid in test_ids], ignore_index=True)
        if len(train_ids) < min_training_matches:
            continue

        # Strict prequential maturity boundary: every training label must have
        # become available no later than the earliest prediction cutoff in the
        # test block. This blocks leakage from simultaneous/overlapping matches
        # whose final label matured after the test prediction point.
        test_start = pd.to_datetime(
            test["prediction_cutoff_utc"], utc=True
        ).min()
        mature_train = train[
            pd.to_datetime(train["label_available_at_utc"], utc=True) <= test_start
        ].copy()
        if mature_train["match_id"].nunique() < min_training_matches:
            continue
        as_of = test_start
        model = fit_hazard_model(
            mature_train,
            as_of_cutoff=as_of,
            method=method,
            min_rows=max(120, min_training_matches * 3),
        )
        x = test[model["feature_columns"]]
        probabilities = model["model"].predict_proba(x)
        model_classes = [str(v) for v in model["model"].classes_]
        aligned = np.full(
            (len(test), len(EVENT_TYPES)),
            1e-9,
            dtype=float,
        )
        for idx, cls in enumerate(model_classes):
            if cls in _EVENT_SET:
                aligned[:, EVENT_TYPES.index(cls)] = probabilities[:, idx]
        aligned /= aligned.sum(axis=1, keepdims=True)
        y = test["next_event_type"].astype(str).to_numpy()

        row_losses = -np.log(
            np.clip(
                aligned[
                    np.arange(len(y)),
                    [EVENT_TYPES.index(v) for v in y],
                ],
                1e-12,
                1.0,
            )
        )
        per_match = (
            pd.DataFrame(
                {"match_id": test["match_id"].astype(str), "loss": row_losses}
            )
            .groupby("match_id", sort=False)["loss"]
            .mean()
        )
        fold_metrics.append({
            "fold": fold_number,
            "train_matches": int(mature_train["match_id"].nunique()),
            "test_matches": int(len(test_ids)),
            "train_snapshots": int(len(mature_train)),
            "test_snapshots": int(len(test)),
            "test_start_utc": str(test_start),
            "test_end_utc": str(pd.to_datetime(test["prediction_cutoff_utc"], utc=True).max()),
            "training_label_cutoff_utc": str(
                pd.to_datetime(mature_train["label_available_at_utc"], utc=True).max()
            ),
            # Primary aggregate is equal-weighted by match, not by snapshot
            # count. Snapshot-weighted loss is retained as a diagnostic only.
            "next_event_logloss": float(per_match.mean()),
            "snapshot_weighted_next_event_logloss": _safe_multiclass_logloss(
                y, aligned, EVENT_TYPES
            ),
        })
        all_y.extend(y.tolist())
        all_p.extend(aligned.tolist())

    if not fold_metrics:
        return {
            "status": "INSUFFICIENT_TRAINING_HISTORY",
            "oos_claimed": False,
            "production_usable": False,
            "folds": [],
        }

    p = np.asarray(all_p, dtype=float)
    y = np.asarray(all_y, dtype=str)
    ll_snapshot = _safe_multiclass_logloss(y, p, EVENT_TYPES)
    ll_match = float(np.mean([f["next_event_logloss"] for f in fold_metrics]))
    return {
        "status": "EVALUATED",
        "method": method,
        "oos_claimed": True,
        "production_usable": False,
        "test_match_count_sum": int(sum(f["test_matches"] for f in fold_metrics)),
        "snapshot_rows": int(len(y)),
        "folds": fold_metrics,
        # Primary: equal-weighted by match to avoid snapshot-count bias.
        "overall_next_event_logloss": ll_match,
        "snapshot_weighted_next_event_logloss": ll_snapshot,
        "policy": (
            "match-level_expanding_chronological_OOS; same-match_snapshots_never_split; "
            "training_labels_must_be_mature_by_test_cutoff"
        ),
    }


def scenario_outcome_from_state_model(
    model: Mapping[str, Any],
    state_frame: pd.DataFrame,
    *,
    horizon_minutes: float | None = None,
    step_minutes: float | None = None,
) -> pd.DataFrame:
    """Generate scenario-derived 1X2 probabilities for each validated snapshot."""
    d = build_state_features(validate_snapshot_contract(state_frame))
    rows = []
    for _, row in d.iterrows():
        result = propagate_scenarios(
            model,
            row.to_dict(),
            horizon_minutes=horizon_minutes,
            step_minutes=step_minutes,
        )
        outcome = result["outcome_probabilities"]
        rows.append({
            "match_id": str(row["match_id"]),
            "prediction_cutoff_utc": str(row["prediction_cutoff_utc"]),
            "home": float(outcome["home"]),
            "draw": float(outcome["draw"]),
            "away": float(outcome["away"]),
            "scenario_state_count": int(result["state_count"]),
            "scenario_pruned_mass": float(result["pruned_mass"]),
        })
    return pd.DataFrame(rows)

def outcome_from_final_scores(home: float, away: float) -> int:
    return 0 if home > away else 1 if home == away else 2


def outcome_metrics(
    y: Sequence[int],
    probabilities: Sequence[Sequence[float]],
) -> dict[str, float]:
    p = np.asarray(probabilities, dtype=float)
    yy = np.asarray(y, dtype=int)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) != len(yy):
        raise ValueError("outcome metric shapes are invalid")
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    ll = -np.log(p[np.arange(len(yy)), yy])
    truth = np.eye(3)[yy]
    brier = np.sum((p - truth) ** 2, axis=1)
    return {
        "n": int(len(yy)),
        "logloss": float(np.mean(ll)),
        "brier": float(np.mean(brier)),
        "accuracy": float(np.mean(np.argmax(p, axis=1) == yy)),
    }


def snapshot_fingerprint(frame: pd.DataFrame) -> str:
    stable_cols = sorted(c for c in frame.columns if c not in {"retrieved_at_utc"})
    stable = frame[stable_cols].copy()
    payload = stable.to_json(
        orient="records",
        date_format="iso",
        double_precision=15,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_match_state_research(
    input_path: str | Path,
    artifact_dir: str | Path,
) -> dict[str, Any]:
    """Run schema/PIT verification without inventing data or performance claims."""
    path = Path(input_path)
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not path.is_file():
        status = {
            "schema_version": 1,
            "status": "WARMUP",
            "reason": "match_state_snapshot_dataset_missing",
            "input_path": str(path),
            "oos_claimed": False,
            "production_usable": False,
            "research_only": True,
            "pit_status": "NOT_EXECUTED",
            "next_step": "acquire_historical_PIT_verified_in_play_snapshots",
        }
        (out / "match_state_status.json").write_text(
            json.dumps(status, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return status

    frame = pd.read_csv(path)
    validated = validate_snapshot_contract(frame)
    enriched = build_state_features(validated)
    fingerprint = snapshot_fingerprint(enriched)
    enriched.to_csv(out / "validated_snapshots.csv", index=False)

    oos_results = {}
    for method in ("logistic", "histgb"):
        try:
            oos_results[method] = evaluate_hazard_chronological_oos(
                enriched,
                method=method,
                min_training_matches=60,
                min_test_matches=10,
                max_folds=8,
            )
        except Exception as exc:
            oos_results[method] = {
                "status": "FAILED",
                "oos_claimed": False,
                "production_usable": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    evaluated = [
        v for v in oos_results.values()
        if isinstance(v, dict) and v.get("status") == "EVALUATED"
    ]
    status = {
        "schema_version": 1,
        "status": "OOS_EVALUATED" if evaluated else "SCHEMA_VERIFIED",
        "rows": int(len(enriched)),
        "matches": int(enriched["match_id"].nunique()),
        "input_fingerprint": fingerprint,
        "oos_claimed": bool(evaluated),
        "performance_verified": False,
        "promotion_candidate": False,
        "production_usable": False,
        "research_only": True,
        "pit_status": "PASS",
        "oos": oos_results,
        "next_step": "calibration_and_robustness_required" if evaluated else "chronological_OOS_required",
    }
    (out / "match_state_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return status


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/research/match_state_snapshots.csv")
    parser.add_argument("--out", default="artifacts/match_state_research")
    args = parser.parse_args()
    print(
        json.dumps(
            run_match_state_research(args.input, args.out),
            ensure_ascii=False,
        )
    )
