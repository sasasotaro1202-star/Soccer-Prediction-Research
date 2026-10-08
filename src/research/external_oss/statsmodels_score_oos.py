"""Research-only chronological OOS runner for the statsmodels Poisson score challenger.

The runner never touches Production artifacts and never evaluates or tunes on a
Frozen Holdout. Each outer fold derives a prediction cutoff from the first OOS
kickoff and uses only rows whose event and source availability precede that cutoff.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.prediction.secondary_outputs import fit_score_rate_model, predict_score_distribution
from src.research.external_oss.statsmodels_poisson import (
    fit_statsmodels_poisson_score_model,
    predict_statsmodels_poisson_distribution,
)


REQUIRED_COLUMNS = {
    "match_id",
    "kickoff_utc",
    "home_team",
    "away_team",
    "competition",
    "home_goals",
    "away_goals",
    "pit_verified",
}


def _binary_ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=int).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if mask.any():
            total += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(total)


def _distribution_metrics(
    block: pd.DataFrame,
    model: Any,
    distribution_fn: Any,
) -> dict[str, float]:
    exact_losses: list[float] = []
    top1_hits = 0
    top3_hits = 0
    top4_hits = 0
    ou_p: list[float] = []
    ou_y: list[int] = []
    btts_p: list[float] = []
    btts_y: list[int] = []

    for row in block.itertuples(index=False):
        dist = distribution_fn(
            model,
            row.home_team,
            row.away_team,
            row.competition,
            max_goals=12,
        )
        dist = sorted(dist, key=lambda x: (-float(x[2]), int(x[0]), int(x[1])))
        actual = (int(row.home_goals), int(row.away_goals))
        lookup = {(int(h), int(a)): float(p) for h, a, p in dist}
        actual_prob = float(np.clip(lookup.get(actual, 0.0), 1e-12, 1.0))
        exact_losses.append(-math.log(actual_prob))
        top1_hits += int(actual == (int(dist[0][0]), int(dist[0][1])))
        top3_hits += int(actual in {(int(h), int(a)) for h, a, _ in dist[:3]})
        top4_hits += int(actual in {(int(h), int(a)) for h, a, _ in dist[:4]})

        total_p = float(
            sum(float(p) for h, a, p in dist if int(h) + int(a) >= 3)
        )
        btts_prob = float(
            sum(float(p) for h, a, p in dist if int(h) >= 1 and int(a) >= 1)
        )
        ou_p.append(total_p)
        ou_y.append(int(actual[0] + actual[1] >= 3))
        btts_p.append(btts_prob)
        btts_y.append(int(actual[0] >= 1 and actual[1] >= 1))

    n = len(block)
    ou_p_arr = np.clip(np.asarray(ou_p, dtype=float), 1e-9, 1.0 - 1e-9)
    btts_p_arr = np.clip(np.asarray(btts_p, dtype=float), 1e-9, 1.0 - 1e-9)
    ou_y_arr = np.asarray(ou_y, dtype=int)
    btts_y_arr = np.asarray(btts_y, dtype=int)

    ou_logloss = float(
        -np.mean(ou_y_arr * np.log(ou_p_arr) + (1 - ou_y_arr) * np.log(1 - ou_p_arr))
    )
    btts_logloss = float(
        -np.mean(
            btts_y_arr * np.log(btts_p_arr)
            + (1 - btts_y_arr) * np.log(1 - btts_p_arr)
        )
    )
    return {
        "n": float(n),
        "score_logloss": float(np.mean(exact_losses)),
        "score_top1": float(top1_hits / n),
        "score_top3": float(top3_hits / n),
        "score_top4": float(top4_hits / n),
        "over_2_5_logloss": ou_logloss,
        "over_2_5_brier": float(np.mean((ou_p_arr - ou_y_arr) ** 2)),
        "over_2_5_ece": _binary_ece(ou_y_arr, ou_p_arr),
        "btts_logloss": btts_logloss,
        "btts_brier": float(np.mean((btts_p_arr - btts_y_arr) ** 2)),
        "btts_ece": _binary_ece(btts_y_arr, btts_p_arr),
    }


def _advance_past_same_kickoff(df: pd.DataFrame, index: int) -> int:
    """Move a boundary past all rows sharing the same kickoff timestamp."""
    boundary = int(index)
    if boundary <= 0 or boundary >= len(df):
        return boundary
    kickoff = df.iloc[boundary - 1]["kickoff_utc"]
    while boundary < len(df) and df.iloc[boundary]["kickoff_utc"] == kickoff:
        boundary += 1
    return boundary


def run_statsmodels_score_oos(
    history: pd.DataFrame,
    *,
    min_train: int = 1000,
    oos_block: int = 2000,
    cutoff_buffer_minutes: int = 60,
    development_end_utc: str | pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Compare statsmodels against the incumbent on chronological OOS only.

    Selection is deliberately absent: this runner reports a fixed challenger
    comparison. Candidate selection/calibration/robustness/holdout decisions
    happen in later gates.
    """
    missing = sorted(REQUIRED_COLUMNS - set(history.columns))
    if missing:
        raise ValueError(f"statsmodels OOS data missing columns: {missing}")

    d = history.copy()
    d["kickoff_utc"] = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    d["home_goals"] = pd.to_numeric(d["home_goals"], errors="coerce")
    d["away_goals"] = pd.to_numeric(d["away_goals"], errors="coerce")
    d["pit_verified"] = d["pit_verified"].astype("boolean")
    d["match_id"] = d["match_id"].astype("string").str.strip()

    if d["match_id"].eq("").any() or d["match_id"].isna().any():
        raise ValueError("statsmodels OOS data contains empty/missing match_id")
    if d["match_id"].duplicated().any():
        raise ValueError("statsmodels OOS data contains duplicate match_id")

    pit_mask = d["pit_verified"].eq(True)

    # Prefer a complete predictor-side replay boundary. Some historical artifacts
    # retain source_available_at_utc as NaT while feature replay has the explicit
    # feature_source_max_available_at_utc populated.
    availability_candidates: list[tuple[str, pd.Series]] = []
    if "feature_source_max_available_at_utc" in d.columns:
        availability_candidates.append(
            (
                "feature_source_max_available_at_utc",
                pd.to_datetime(
                    d["feature_source_max_available_at_utc"],
                    utc=True,
                    errors="coerce",
                ),
            )
        )
    if "source_available_at_utc" in d.columns:
        availability_candidates.append(
            (
                "source_available_at_utc",
                pd.to_datetime(
                    d["source_available_at_utc"],
                    utc=True,
                    errors="coerce",
                ),
            )
        )

    availability_series = None
    availability_name = None
    for name, series in availability_candidates:
        if pit_mask.any() and bool(series.loc[pit_mask].notna().all()):
            availability_name = name
            availability_series = series
            break
    if availability_series is None:
        raise ValueError(
            "statsmodels OOS data has no complete explicit predictor-side availability "
            "timestamp for all PIT-verified rows"
        )

    d["source_available_at_utc"] = availability_series

    if d.loc[pit_mask, "kickoff_utc"].isna().any() or d.loc[pit_mask, "source_available_at_utc"].isna().any():
        raise ValueError(
            "statsmodels OOS data contains unknown timestamps in PIT-verified rows"
        )
    if d.loc[pit_mask, "competition"].astype(str).str.strip().eq("").any():
        raise ValueError(
            "statsmodels OOS data contains empty competition in PIT-verified rows"
        )
    if (d.loc[pit_mask, ["home_goals", "away_goals"]] < 0).any().any():
        raise ValueError(
            "statsmodels OOS goal labels must be non-negative in PIT-verified rows"
        )

    d = d[
        pit_mask
        & d["kickoff_utc"].notna()
        & d["source_available_at_utc"].notna()
        & d["home_goals"].notna()
        & d["away_goals"].notna()
    ].sort_values(["kickoff_utc", "match_id"], kind="mergesort").reset_index(drop=True)

    if development_end_utc is not None:
        development_end = pd.to_datetime(
            development_end_utc, utc=True, errors="coerce"
        )
        if pd.isna(development_end):
            raise ValueError("development_end_utc is invalid")
        d = d[d["kickoff_utc"] < development_end].copy()

    if len(d) < int(min_train) + int(oos_block):
        raise ValueError(
            f"Not enough PIT-verified rows for OOS: {len(d)}; "
            f"need at least {int(min_train) + int(oos_block)}"
        )

    rows: list[dict[str, Any]] = []
    start = _advance_past_same_kickoff(d, int(min_train))
    fold = 0
    while start < len(d):
        # Derive the cutoff from the first OOS event, then verify the training
        # prefix after applying the same PIT boundary. If availability filtering
        # removes one or more rows, move the OOS boundary forward rather than
        # lowering the minimum training requirement.
        prediction_cutoff = d.iloc[start]["kickoff_utc"] - pd.Timedelta(
            minutes=int(cutoff_buffer_minutes)
        )
        train = d.iloc[:start].copy()
        train = train[
            (train["kickoff_utc"] < prediction_cutoff)
            & (train["source_available_at_utc"] <= prediction_cutoff)
        ].copy()

        if len(train) < int(min_train):
            next_start = _advance_past_same_kickoff(d, start + 1)
            if next_start <= start:
                raise ValueError(
                    f"Fold {fold}: unable to establish a PIT-safe training prefix "
                    f"with at least {int(min_train)} rows"
                )
            start = next_start
            continue

        end = _advance_past_same_kickoff(
            d, min(start + int(oos_block), len(d))
        )
        oos = d.iloc[start:end].copy()
        if oos.empty:
            break

        incumbent = fit_score_rate_model(train)
        challenger = fit_statsmodels_poisson_score_model(
            train, prediction_cutoff_utc=prediction_cutoff
        )

        # Fair-comparison universe: only score the cases both fixed models can
        # represent from the same training prefix. Unsupported teams/competitions
        # are retained as explicit coverage/OOD exclusions, never imputed.
        incumbent_teams = set((incumbent.get("teams") or {}).keys())
        challenger_home = set(challenger.categories_home_team)
        challenger_away = set(challenger.categories_away_team)
        challenger_competitions = set(challenger.categories_competition)

        eligibility = (
            oos["home_team"].astype("string").isin(incumbent_teams)
            & oos["away_team"].astype("string").isin(incumbent_teams)
            & oos["home_team"].astype("string").isin(challenger_home)
            & oos["away_team"].astype("string").isin(challenger_away)
            & oos["competition"].astype("string").isin(challenger_competitions)
        )
        common_oos = oos.loc[eligibility].copy()
        excluded_oos = oos.loc[~eligibility].copy()

        if common_oos.empty:
            raise ValueError(
                f"Fold {fold}: no common evaluable OOS cases after explicit "
                "model-support filtering"
            )

        base_metrics = _distribution_metrics(
            common_oos, incumbent, predict_score_distribution
        )
        candidate_metrics = _distribution_metrics(
            common_oos, challenger, predict_statsmodels_poisson_distribution
        )

        rows.append(
            {
                "fold": int(fold),
                "oos_start_utc": str(oos["kickoff_utc"].min()),
                "oos_end_utc": str(oos["kickoff_utc"].max()),
                "prediction_cutoff_utc": str(prediction_cutoff),
                "training_rows": int(len(train)),
                "oos_rows": int(len(oos)),
                "common_evaluable_rows": int(len(common_oos)),
                "unsupported_rows": int(len(excluded_oos)),
                "common_coverage": float(len(common_oos) / len(oos)),
                "incumbent_score_logloss": base_metrics["score_logloss"],
                "statsmodels_score_logloss": candidate_metrics["score_logloss"],
                "incumbent_score_top1": base_metrics["score_top1"],
                "statsmodels_score_top1": candidate_metrics["score_top1"],
                "incumbent_score_top3": base_metrics["score_top3"],
                "statsmodels_score_top3": candidate_metrics["score_top3"],
                "incumbent_score_top4": base_metrics["score_top4"],
                "statsmodels_score_top4": candidate_metrics["score_top4"],
                "incumbent_over_2_5_logloss": base_metrics["over_2_5_logloss"],
                "statsmodels_over_2_5_logloss": candidate_metrics["over_2_5_logloss"],
                "incumbent_over_2_5_brier": base_metrics["over_2_5_brier"],
                "statsmodels_over_2_5_brier": candidate_metrics["over_2_5_brier"],
                "incumbent_over_2_5_ece": base_metrics["over_2_5_ece"],
                "statsmodels_over_2_5_ece": candidate_metrics["over_2_5_ece"],
                "incumbent_btts_logloss": base_metrics["btts_logloss"],
                "statsmodels_btts_logloss": candidate_metrics["btts_logloss"],
                "incumbent_btts_brier": base_metrics["btts_brier"],
                "statsmodels_btts_brier": candidate_metrics["btts_brier"],
                "incumbent_btts_ece": base_metrics["btts_ece"],
                "statsmodels_btts_ece": candidate_metrics["btts_ece"],
                "selection_performed": False,
                "frozen_holdout_used": False,
                "production_usable": False,
                "pit_training_boundary_valid": bool(
                    (train["kickoff_utc"] < prediction_cutoff).all()
                    and (train["source_available_at_utc"] <= prediction_cutoff).all()
                ),
                "same_kickoff_split_avoided": bool(
                    train["kickoff_utc"].max() < oos["kickoff_utc"].min()
                ),
            }
        )
        fold += 1
        start = end

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("statsmodels OOS produced no evaluation folds")

    return {
        "schema_version": 1,
        "status": "RESEARCH_OOS_READY",
        "research_only": True,
        "production_usable": False,
        "selection_performed": False,
        "frozen_holdout_used": False,
        "fold_count": int(len(result)),
        "rows": result.to_dict(orient="records"),
    }
