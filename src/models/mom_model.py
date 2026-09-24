"""PIT-safe Man of the Match ranking candidate.

This module is research-only until its chronological OOS, calibration, robustness,
and frozen-holdout gates are satisfied. It deliberately separates player eligibility
from the MOM ranking model: candidate rows must already be known before kickoff.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.mom_selector import MOMCandidate, select_mom_candidates


MOM_FEATURE_COLUMNS = (
    "recent_rating_ewm",
    "recent_minutes_ewm",
    "recent_goals_per90_ewm",
    "recent_assists_per90_ewm",
    "recent_xg_per90_ewm",
    "recent_xa_per90_ewm",
    "recent_key_passes_per90_ewm",
    "recent_shots_per90_ewm",
    "recent_starts_rate",
    "team_attack_strength",
    "opponent_defense_strength",
    "position_attack_weight",
    "days_rest",
)

MOM_REQUIRED_COLUMNS = {
    "match_id",
    "player_id",
    "kickoff_utc",
    "feature_available_at_utc",
    "pit_verified",
}

_EPS = 1e-12


@dataclass(frozen=True)
class MOMModelMetadata:
    schema_version: int
    method: str
    feature_columns: tuple[str, ...]
    training_rows: int
    training_matches: int
    positive_rows: int
    negative_rows: int
    temperature: float


def _parse_utc(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if values.isna().any():
        raise ValueError(f"MOM column {column!r} contains invalid or missing timestamps")
    return values


def _strict_bool(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        if values.isna().any():
            raise ValueError(f"MOM column {column!r} contains missing booleans")
        return values.astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    valid = normalized.isin({"true", "false", "1", "0", "yes", "no"})
    if not bool(valid.all()):
        raise ValueError(f"MOM column {column!r} contains non-boolean values")
    return normalized.isin({"true", "1", "yes"})


def _validate_base(frame: pd.DataFrame, *, prediction: bool = False) -> pd.DataFrame:
    missing = sorted(MOM_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"MOM data missing required columns: {missing}")

    d = frame.copy()
    d["match_id"] = d["match_id"].astype("string").str.strip()
    d["player_id"] = d["player_id"].astype("string").str.strip()
    if d["match_id"].isna().any() or d["match_id"].eq("").any():
        raise ValueError("MOM data contains missing/empty match_id")
    if d["player_id"].isna().any() or d["player_id"].eq("").any():
        raise ValueError("MOM data contains missing/empty player_id")
    d["kickoff_utc"] = _parse_utc(d, "kickoff_utc")
    d["feature_available_at_utc"] = _parse_utc(d, "feature_available_at_utc")
    d["pit_verified"] = _strict_bool(d, "pit_verified")

    # A feature used for a match can never be available after that match kickoff.
    bad_pit = d["feature_available_at_utc"] > d["kickoff_utc"]
    if bool(bad_pit.any()):
        raise RuntimeError("MOM PIT violation: feature_available_at_utc is after kickoff_utc")

    for column in MOM_FEATURE_COLUMNS:
        if column not in d.columns:
            raise ValueError(f"MOM data missing feature column {column!r}")
        d[column] = pd.to_numeric(d[column], errors="coerce")

    bad_numeric = ~np.isfinite(d[list(MOM_FEATURE_COLUMNS)].to_numpy(dtype=float)).all(axis=1)
    if bool(bad_numeric.any()):
        raise ValueError("MOM feature matrix contains NaN, inf, or non-numeric values")

    if d.duplicated(subset=["match_id", "player_id"]).any():
        raise ValueError("MOM data contains duplicate match_id/player_id rows")

    if not prediction:
        if "is_motm" not in d.columns:
            raise ValueError("MOM training data requires is_motm")
        labels = pd.to_numeric(d["is_motm"], errors="coerce")
        if labels.isna().any() or ~labels.isin([0, 1]).all():
            raise ValueError("is_motm must contain only 0/1")
        d["is_motm"] = labels.astype(int)
        counts = d.groupby("match_id", sort=False)["is_motm"].sum()
        if bool((counts != 1).any()):
            bad = counts[counts != 1].head(10).to_dict()
            raise ValueError(f"Each MOM training match must have exactly one positive label: {bad}")
    return d


def _verified_training_slice(history: pd.DataFrame) -> pd.DataFrame:
    d = _validate_base(history, prediction=False)
    # Unknown/unverified PIT rows are unusable. They are excluded rather than
    # imputed or treated as safe. The caller still gets a hard failure if nothing
    # remains after this gate.
    d = d.loc[d["pit_verified"]].copy()
    if d.empty:
        raise RuntimeError("No PIT-verified MOM training rows remain")
    d = d.sort_values(["kickoff_utc", "match_id", "player_id"], kind="mergesort").reset_index(drop=True)
    return d


def fit_mom_model(
    history: pd.DataFrame,
    *,
    regularization_c: float = 0.30,
    temperature: float = 1.0,
    min_matches: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit a conservative binary MOM scorer on PIT-verified historical rows.

    OOS selection/calibration must be performed outside this function on chronological
    slices. The final frozen holdout must never be passed into this fit call.
    """
    if not np.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and > 0")
    if not np.isfinite(float(regularization_c)) or float(regularization_c) <= 0:
        raise ValueError("regularization_c must be finite and > 0")

    d = _verified_training_slice(history)
    match_count = int(d["match_id"].nunique())
    if match_count < int(min_matches):
        raise RuntimeError(f"Insufficient PIT-verified MOM matches: {match_count} < {min_matches}")

    y = d["is_motm"].to_numpy(dtype=int)
    if np.unique(y).size != 2:
        raise RuntimeError("MOM training requires both positive and negative classes")

    model = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=3000,
            C=float(regularization_c),
            class_weight="balanced",
            random_state=random_state,
        )),
    ])
    model.fit(d[list(MOM_FEATURE_COLUMNS)], y)

    meta = MOMModelMetadata(
        schema_version=1,
        method="pit_player_form_logistic_softmax",
        feature_columns=MOM_FEATURE_COLUMNS,
        training_rows=int(len(d)),
        training_matches=match_count,
        positive_rows=int(y.sum()),
        negative_rows=int((y == 0).sum()),
        temperature=float(temperature),
    )
    return {"model": model, "metadata": meta.__dict__.copy()}


def predict_mom_distribution(
    model_bundle: dict[str, Any],
    candidates: pd.DataFrame,
    *,
    prediction_time: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return a full probability distribution over eligible player candidates."""
    if not isinstance(model_bundle, dict) or "model" not in model_bundle:
        raise ValueError("MOM model bundle is invalid")
    d = _validate_base(candidates, prediction=True)
    if "is_motm" in d.columns:
        raise ValueError("MOM prediction input must not contain future outcome labels")

    model = model_bundle["model"]
    temperature = float((model_bundle.get("metadata") or {}).get("temperature", 1.0))
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("MOM model temperature is invalid")

    if prediction_time is not None:
        pt = pd.Timestamp(prediction_time)
        if pt.tzinfo is None:
            pt = pt.tz_localize("UTC")
        else:
            pt = pt.tz_convert("UTC")
        if bool((d["feature_available_at_utc"] > pt).any()):
            raise RuntimeError("MOM prediction contains a feature unavailable at prediction_time")
        if bool((d["kickoff_utc"] <= pt).any()):
            raise RuntimeError("MOM prediction contains a fixture that is not in the future")

    # A player candidate is required to be PIT-safe at the time of prediction.
    d = d.loc[d["pit_verified"]].copy()
    if d.empty:
        raise RuntimeError("No PIT-verified MOM candidates remain")

    logits = np.asarray(model.decision_function(d[list(MOM_FEATURE_COLUMNS)]), dtype=float).reshape(-1)
    if not np.isfinite(logits).all():
        raise RuntimeError("MOM model produced non-finite logits")

    # Normalize only within each fixture. A single global softmax across multiple
    # fixtures would make probabilities depend on which other fixtures were batched.
    logits = logits / temperature
    probabilities = np.zeros(len(d), dtype=float)
    for match_id, idx in d.groupby("match_id", sort=False).groups.items():
        loc = np.asarray(list(idx), dtype=int)
        local_logits = logits[loc]
        local_logits = local_logits - np.max(local_logits)
        exp_logits = np.exp(np.clip(local_logits, -700.0, 700.0))
        denom = max(float(exp_logits.sum()), _EPS)
        probabilities[loc] = exp_logits / denom
        if not np.isfinite(probabilities[loc]).all() or abs(float(probabilities[loc].sum()) - 1.0) > 1e-9:
            raise RuntimeError(f"MOM probability distribution is invalid for match_id={match_id!r}")

    out = d[["match_id", "player_id", "kickoff_utc"]].copy()
    out["probability"] = probabilities
    return out.sort_values(["match_id", "probability", "player_id"], ascending=[True, False, True], kind="mergesort").reset_index(drop=True)


def select_mom_top4_from_distribution(distribution: pd.DataFrame) -> list[MOMCandidate]:
    """Select exactly four candidates from exactly one fixture's full distribution."""
    required = {"match_id", "player_id", "probability"}
    if not required.issubset(distribution.columns):
        raise ValueError("MOM distribution missing match_id/player_id/probability")
    match_ids = distribution["match_id"].astype("string").drop_duplicates()
    if len(match_ids) != 1:
        raise ValueError("MOM top-4 selection requires exactly one match_id")
    return select_mom_candidates(distribution["player_id"], distribution["probability"], top_k=4)
