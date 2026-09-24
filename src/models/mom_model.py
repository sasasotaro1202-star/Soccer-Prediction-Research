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
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.mom_selector import MOMCandidate, select_mom_candidates


MOM_FEATURE_COLUMNS = (
    "recent_rating_ewm",
    "recent_minutes_ewm",
    "recent_goals_per90_ewm",
    "recent_assists_per90_ewm",
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
    missingness_aware_imputation: bool


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
    bad_pit = d["feature_available_at_utc"] >= d["kickoff_utc"]
    if bool(bad_pit.any()):
        raise RuntimeError("MOM PIT violation: feature_available_at_utc is not strictly before kickoff_utc")

    for column in MOM_FEATURE_COLUMNS:
        if column not in d.columns:
            raise ValueError(f"MOM data missing feature column {column!r}")
        raw = d[column]
        converted = pd.to_numeric(raw, errors="coerce")
        invalid = raw.notna() & converted.isna()
        if bool(invalid.any()):
            raise ValueError(f"MOM feature column {column!r} contains non-numeric values")
        if bool(np.isinf(converted.to_numpy(dtype=float)).any()):
            raise ValueError(f"MOM feature column {column!r} contains infinite values")
        d[column] = converted

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




class MissingnessAwareMOMTransformer:
    """Median-impute historical-only features while preserving missingness flags.

    Medians are learned strictly from the fit slice. Missing values are never
    interpreted as zero performance; a binary missingness indicator is appended
    for every model feature.
    """

    def fit(self, frame: pd.DataFrame, y: Any = None) -> "MissingnessAwareMOMTransformer":
        x = frame[list(MOM_FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
        self.medians_ = x.median(axis=0, skipna=True).to_numpy(dtype=float)
        if not np.isfinite(self.medians_).all():
            bad = [c for c, v in zip(MOM_FEATURE_COLUMNS, self.medians_) if not np.isfinite(v)]
            raise RuntimeError(f"MOM imputation cannot learn medians for all-missing features: {bad}")
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "medians_"):
            raise RuntimeError("MOM imputer is not fitted")
        x = frame[list(MOM_FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
        values = x.to_numpy(dtype=float)
        missing = ~np.isfinite(values)
        filled = np.where(missing, self.medians_[None, :], values)
        out = np.concatenate([filled, missing.astype(float)], axis=1)
        if not np.isfinite(out).all():
            raise RuntimeError("MOM imputation produced non-finite features")
        return out

    def fit_transform(self, frame: pd.DataFrame, y: Any = None) -> np.ndarray:
        return self.fit(frame, y).transform(frame)


class ConditionalMOMLogit:
    """Conditional-choice model: exactly one MOM winner per fixture.

    The objective is a grouped conditional log-likelihood, so class imbalance
    from varying squad sizes is handled by the softmax denominator rather than
    an arbitrary binary class weight.
    """

    def __init__(self, l2: float = 1.0):
        self.l2 = float(l2)

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> "ConditionalMOMLogit":
        labels = np.asarray(y, dtype=int)
        self.transformer_ = MissingnessAwareMOMTransformer()
        x = self.transformer_.fit_transform(frame[list(MOM_FEATURE_COLUMNS)])
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        z = (x - self.mean_) / self.scale_

        # Caller supplies rows sorted by match. Each contiguous group has exactly
        # one positive label; this makes the optimization a grouped conditional logit.
        match_ids = frame.index.to_numpy()
        groups = []
        start = 0
        if len(z):
            current = match_ids[0]
            for i, value in enumerate(match_ids[1:], start=1):
                if value != current:
                    groups.append((start, i))
                    start = i
                    current = value
            groups.append((start, len(z)))

        def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
            loss = 0.5 * self.l2 * float(np.dot(beta, beta))
            grad = self.l2 * beta.copy()
            for lo, hi in groups:
                zg = z[lo:hi]
                yg = labels[lo:hi]
                positive = np.flatnonzero(yg == 1)
                if len(positive) != 1:
                    raise RuntimeError("Conditional MOM training requires exactly one positive per match")
                scores = zg @ beta
                shift = float(np.max(scores))
                exp_scores = np.exp(np.clip(scores - shift, -700.0, 700.0))
                probs = exp_scores / max(float(exp_scores.sum()), _EPS)
                target = positive[0]
                loss += -float(scores[target] - shift - np.log(max(float(exp_scores.sum()), _EPS)))
                grad += zg.T @ probs
                grad -= zg[target]
            return float(loss), grad

        result = minimize(
            lambda beta: objective(beta)[0],
            np.zeros(z.shape[1], dtype=float),
            jac=lambda beta: objective(beta)[1],
            method="L-BFGS-B",
            options={"maxiter": 400, "ftol": 1e-10, "gtol": 1e-7},
        )
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"Conditional MOM optimization failed: {result.message}")
        self.coef_ = np.asarray(result.x, dtype=float)
        return self

    def decision_function(self, frame: pd.DataFrame) -> np.ndarray:
        x = self.transformer_.transform(frame[list(MOM_FEATURE_COLUMNS)])
        z = (x - self.mean_) / self.scale_
        scores = z @ self.coef_
        if not np.isfinite(scores).all():
            raise RuntimeError("Conditional MOM model produced non-finite scores")
        return scores


def fit_mom_model(
    history: pd.DataFrame,
    *,
    regularization_c: float = 0.30,
    temperature: float = 1.0,
    min_matches: int = 5,
    random_state: int = 42,
    method: str = "binary_logit",
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

    if method == "conditional_logit":
        # Preserve chronological row/group order; ConditionalMOMLogit consumes
        # complete fixture groups and models the winner choice directly.
        model = ConditionalMOMLogit(l2=1.0 / max(float(regularization_c), 1e-9))
        ordered = d.sort_values(["kickoff_utc", "match_id", "player_id"], kind="mergesort").reset_index(drop=True)
        model.fit(ordered[list(MOM_FEATURE_COLUMNS)].set_axis(ordered["match_id"].astype(str).to_numpy(), axis="index"), ordered["is_motm"].to_numpy(dtype=int))
        d = ordered
    elif method == "binary_logit":
        model = Pipeline([
            ("impute_missingness", MissingnessAwareMOMTransformer()),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000,
                C=float(regularization_c),
                class_weight="balanced",
                random_state=random_state,
            )),
        ])
        model.fit(d[list(MOM_FEATURE_COLUMNS)], y)
    else:
        raise ValueError("Unsupported MOM training method")

    meta = MOMModelMetadata(
        schema_version=1,
        method="pit_player_form_conditional_logit" if method == "conditional_logit" else "pit_player_form_logistic_softmax",
        feature_columns=MOM_FEATURE_COLUMNS,
        training_rows=int(len(d)),
        training_matches=match_count,
        positive_rows=int(y.sum()),
        negative_rows=int((y == 0).sum()),
        temperature=float(temperature),
        missingness_aware_imputation=True,
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

    # Group-local probability arrays use positional indices. Resetting here prevents
    # a sliced OOS frame's original labels from becoming out-of-bounds NumPy indices.
    d = d.reset_index(drop=True)

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
        if bool((d["feature_available_at_utc"] >= d["kickoff_utc"]).any()):
            raise RuntimeError("MOM prediction PIT violation: feature unavailable at or after kickoff")
        if bool((d["feature_available_at_utc"] > pt).any()):
            raise RuntimeError("MOM prediction feature unavailable at prediction_time")
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
