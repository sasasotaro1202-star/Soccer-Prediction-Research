"""Research-only chronological OOS/WFO runner for the statsmodels Poisson score challenger.

This runner is intentionally outside Production authority. It:
- requires explicit TESTS_PASSED and AUDIT_PASSED handoff;
- enforces predictor-side PIT at the case cutoff;
- fits model parameters only on the chronological training side;
- calibrates only on a later chronological slice inside training;
- scores only a common model-support OOS universe;
- records unsupported/OOD cases explicitly;
- never touches the Frozen Holdout or Production bundle;
- never performs candidate selection.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from src.prediction.secondary_outputs import (
    fit_score_rate_model,
    predict_score_distribution,
)
from src.research.external_oss.score_distribution_calibration import (
    fit_temperature,
    temperature_transform,
)
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
            total += float(mask.mean()) * abs(
                float(p[mask].mean()) - float(y[mask].mean())
            )
    return float(total)


def _distribution_metrics(
    block: pd.DataFrame,
    model: Any,
    distribution_fn: Callable[..., Any],
) -> dict[str, float]:
    if block.empty:
        raise ValueError("score metric block is empty")

    exact_losses: list[float] = []
    score_briers: list[float] = []
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
        if not dist:
            raise RuntimeError("empty score distribution during OOS evaluation")

        actual = (int(row.home_goals), int(row.away_goals))
        lookup = {(int(h), int(a)): float(p) for h, a, p in dist}
        actual_prob = float(np.clip(lookup.get(actual, 0.0), 1e-12, 1.0))
        exact_losses.append(-math.log(actual_prob))
        score_briers.append(
            float(
                sum(
                    (float(p) - (1.0 if (int(h), int(a)) == actual else 0.0)) ** 2
                    for h, a, p in dist
                )
            )
        )

        top1_hits += int(actual == (int(dist[0][0]), int(dist[0][1])))
        top3_hits += int(actual in {(int(h), int(a)) for h, a, _ in dist[:3]})
        top4_hits += int(actual in {(int(h), int(a)) for h, a, _ in dist[:4]})

        p_over25 = float(sum(float(p) for h, a, p in dist if int(h) + int(a) >= 3))
        p_btts = float(sum(float(p) for h, a, p in dist if int(h) >= 1 and int(a) >= 1))
        ou_p.append(p_over25)
        ou_y.append(int(actual[0] + actual[1] >= 3))
        btts_p.append(p_btts)
        btts_y.append(int(actual[0] >= 1 and actual[1] >= 1))

    n = len(block)
    ou_p_arr = np.clip(np.asarray(ou_p, dtype=float), 1e-9, 1.0 - 1e-9)
    btts_p_arr = np.clip(np.asarray(btts_p, dtype=float), 1e-9, 1.0 - 1e-9)
    ou_y_arr = np.asarray(ou_y, dtype=int)
    btts_y_arr = np.asarray(btts_y, dtype=int)

    ou_logloss = float(
        -np.mean(
            ou_y_arr * np.log(ou_p_arr)
            + (1 - ou_y_arr) * np.log(1 - ou_p_arr)
        )
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
        "score_brier": float(np.mean(score_briers)),
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


def _calibrated_distribution_fn(
    raw_distribution_fn: Callable[..., Any],
    temperature: float,
) -> Callable[..., Any]:
    def predict(
        model: Any,
        home_team: str,
        away_team: str,
        competition: str | None,
        *,
        max_goals: int = 12,
    ) -> list[tuple[int, int, float]]:
        raw = raw_distribution_fn(
            model,
            home_team,
            away_team,
            competition,
            max_goals=max_goals,
        )
        return temperature_transform(raw, temperature)

    return predict


def _support_mask(
    frame: pd.DataFrame,
    *,
    incumbent: Mapping[str, Any],
    challenger: Any,
) -> pd.Series:
    incumbent_teams = set((incumbent.get("teams") or {}).keys())
    challenger_home = set(challenger.categories_home_team)
    challenger_away = set(challenger.categories_away_team)
    challenger_competitions = set(challenger.categories_competition)
    return (
        frame["home_team"].astype("string").isin(incumbent_teams)
        & frame["away_team"].astype("string").isin(incumbent_teams)
        & frame["home_team"].astype("string").isin(challenger_home)
        & frame["away_team"].astype("string").isin(challenger_away)
        & frame["competition"].astype("string").isin(challenger_competitions)
    )


def _as_bool(value: Any) -> bool:
    return type(value) is bool and value is True


def _handoff_status(
    tests_passed: Any,
    audit_passed: Any,
) -> dict[str, Any]:
    if not _as_bool(tests_passed):
        return {
            "status": "BLOCKED",
            "reason": "TESTS_PASSED handoff is missing_or_false",
        }
    if not _as_bool(audit_passed):
        return {
            "status": "BLOCKED",
            "reason": "AUDIT_PASSED handoff is missing_or_false",
        }
    return {"status": "PASS", "reason": "explicit_test_and_audit_handoff_verified"}


def _experiment_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_statsmodels_score_oos(
    history: pd.DataFrame,
    *,
    min_train: int = 1000,
    oos_block: int = 2000,
    cutoff_buffer_minutes: int = 60,
    calibration_fraction: float = 0.20,
    calibration_min_rows: int = 100,
    development_end_utc: str | pd.Timestamp | None = None,
    tests_passed: bool | None = None,
    audit_passed: bool | None = None,
) -> dict[str, Any]:
    """Compare statsmodels against the incumbent on chronological OOS only.

    This function performs no candidate selection and has no Production or
    Frozen Holdout write path.
    """
    handoff = _handoff_status(tests_passed, audit_passed)
    if handoff["status"] != "PASS":
        return {
            "schema_version": 2,
            "status": "BLOCKED",
            "research_only": True,
            "production_usable": False,
            "selection_performed": False,
            "frozen_holdout_used": False,
            "handoff": handoff,
        }

    if int(min_train) < 200:
        raise ValueError("min_train must be at least 200")
    if int(oos_block) < 8:
        raise ValueError("oos_block must be at least 8")
    if not 0.10 <= float(calibration_fraction) <= 0.40:
        raise ValueError("calibration_fraction must be between 0.10 and 0.40")
    if int(calibration_min_rows) < 50:
        raise ValueError("calibration_min_rows must be at least 50")

    missing = sorted(REQUIRED_COLUMNS - set(history.columns))
    if missing:
        raise ValueError(f"statsmodels OOS data missing columns: {missing}")

    d = history.copy()
    d["kickoff_utc"] = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    d["home_goals"] = pd.to_numeric(d["home_goals"], errors="coerce")
    d["away_goals"] = pd.to_numeric(d["away_goals"], errors="coerce")
    d["pit_verified"] = d["pit_verified"].astype("boolean")
    d["match_id"] = d["match_id"].astype("string").str.strip()
    d["home_team"] = d["home_team"].astype("string").str.strip()
    d["away_team"] = d["away_team"].astype("string").str.strip()
    d["competition"] = d["competition"].astype("string").str.strip()

    if d["match_id"].isna().any() or d["match_id"].eq("").any():
        raise ValueError("statsmodels OOS data contains empty/missing match_id")
    if d["match_id"].duplicated().any():
        raise ValueError("statsmodels OOS data contains duplicate match_id")
    if d["home_team"].isna().any() or d["home_team"].eq("").any():
        raise ValueError("statsmodels OOS data contains empty/missing home_team")
    if d["away_team"].isna().any() or d["away_team"].eq("").any():
        raise ValueError("statsmodels OOS data contains empty/missing away_team")
    if d["competition"].isna().any() or d["competition"].eq("").any():
        raise ValueError("statsmodels OOS data contains empty/missing competition")

    pit_mask = d["pit_verified"].eq(True)

    if "feature_source_max_available_at_utc" not in d.columns:
        raise ValueError(
            "statsmodels OOS data requires feature_source_max_available_at_utc "
            "as the predictor-side PIT authority"
        )

    availability_name = "feature_source_max_available_at_utc"
    availability_series = pd.to_datetime(
        d["feature_source_max_available_at_utc"],
        utc=True,
        errors="coerce",
    )
    if pit_mask.any() and not bool(availability_series.loc[pit_mask].notna().all()):
        raise ValueError(
            "statsmodels OOS data contains unknown predictor-side feature availability timestamps"
        )

    d["feature_source_max_available_at_utc"] = availability_series
    if d.loc[pit_mask, "kickoff_utc"].isna().any():
        raise ValueError("statsmodels OOS data contains unknown kickoff timestamps")
    if d.loc[pit_mask, "feature_source_max_available_at_utc"].isna().any():
        raise ValueError(
            "statsmodels OOS data contains unknown predictor-side feature availability timestamps"
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
        prediction_cutoff = d.iloc[start]["kickoff_utc"] - pd.Timedelta(
            minutes=int(cutoff_buffer_minutes)
        )
        train = d.iloc[:start].copy()
        train_own_cutoff = train["kickoff_utc"] - pd.Timedelta(
            minutes=int(cutoff_buffer_minutes)
        )
        train = train[
            (train["kickoff_utc"] < prediction_cutoff)
            & (train["feature_source_max_available_at_utc"] <= train_own_cutoff)
            & (train["feature_source_max_available_at_utc"] <= prediction_cutoff)
        ].copy()

        if len(train) < int(min_train):
            next_start = _advance_past_same_kickoff(d, start + 1)
            if next_start <= start:
                raise ValueError(
                    f"Fold {fold}: unable to establish PIT-safe training prefix "
                    f"with at least {int(min_train)} rows"
                )
            start = next_start
            continue

        end = _advance_past_same_kickoff(d, min(start + int(oos_block), len(d)))
        oos = d.iloc[start:end].copy()
        if oos.empty:
            break

        # OOS PIT is case-specific: each predictor must be available before that
        # match's own prediction cutoff, not merely before the first OOS cutoff.
        oos_case_cutoff = oos["kickoff_utc"] - pd.Timedelta(
            minutes=int(cutoff_buffer_minutes)
        )
        oos_pit_valid = (
            oos["feature_source_max_available_at_utc"] <= oos_case_cutoff
        )
        if not bool(oos_pit_valid.all()):
            raise ValueError(
                f"Fold {fold}: {int((~oos_pit_valid).sum())} OOS rows violate "
                "case-level predictor PIT"
            )

        calibration_rows = max(
            int(calibration_min_rows),
            int(math.ceil(len(train) * float(calibration_fraction))),
        )
        if len(train) <= calibration_rows:
            raise ValueError(
                f"Fold {fold}: training prefix {len(train)} is too small for "
                f"{calibration_rows} calibration rows"
            )

        model_fit = train.iloc[:-calibration_rows].copy()
        calibration = train.iloc[-calibration_rows:].copy()

        incumbent = fit_score_rate_model(model_fit)
        challenger = fit_statsmodels_poisson_score_model(
            model_fit,
            prediction_cutoff_utc=prediction_cutoff,
            regularization_alpha=0.1,
        )

        calibration_common = calibration.loc[
            _support_mask(
                calibration,
                incumbent=incumbent,
                challenger=challenger,
            )
        ].copy()
        if len(calibration_common) < int(calibration_min_rows):
            raise ValueError(
                f"Fold {fold}: only {len(calibration_common)} common-supported "
                f"calibration rows; need at least {int(calibration_min_rows)}"
            )

        incumbent_temperature_info = fit_temperature(
            calibration_common,
            incumbent,
            predict_score_distribution,
        )
        challenger_temperature_info = fit_temperature(
            calibration_common,
            challenger,
            predict_statsmodels_poisson_distribution,
        )
        incumbent_temperature = float(incumbent_temperature_info["temperature"])
        challenger_temperature = float(challenger_temperature_info["temperature"])

        incumbent_calibrated_fn = _calibrated_distribution_fn(
            predict_score_distribution,
            incumbent_temperature,
        )
        challenger_calibrated_fn = _calibrated_distribution_fn(
            predict_statsmodels_poisson_distribution,
            challenger_temperature,
        )

        common_support = _support_mask(
            oos,
            incumbent=incumbent,
            challenger=challenger,
        )
        common_oos = oos.loc[common_support].copy()
        excluded_oos = oos.loc[~common_support].copy()
        if common_oos.empty:
            raise ValueError(
                f"Fold {fold}: no common evaluable OOS cases after explicit "
                "model-support filtering"
            )

        raw_base = _distribution_metrics(
            common_oos,
            incumbent,
            predict_score_distribution,
        )
        raw_candidate = _distribution_metrics(
            common_oos,
            challenger,
            predict_statsmodels_poisson_distribution,
        )
        cal_base = _distribution_metrics(
            common_oos,
            incumbent,
            incumbent_calibrated_fn,
        )
        cal_candidate = _distribution_metrics(
            common_oos,
            challenger,
            challenger_calibrated_fn,
        )

        competition_slices: list[dict[str, Any]] = []
        for competition, comp_block in common_oos.groupby("competition", sort=True):
            comp_base = _distribution_metrics(
                comp_block,
                incumbent,
                incumbent_calibrated_fn,
            )
            comp_candidate = _distribution_metrics(
                comp_block,
                challenger,
                challenger_calibrated_fn,
            )
            competition_slices.append(
                {
                    "competition": str(competition),
                    "n": int(len(comp_block)),
                    "incumbent_calibrated_score_logloss": comp_base["score_logloss"],
                    "statsmodels_calibrated_score_logloss": comp_candidate["score_logloss"],
                    "incumbent_calibrated_score_top3": comp_base["score_top3"],
                    "statsmodels_calibrated_score_top3": comp_candidate["score_top3"],
                    "incumbent_calibrated_over_2_5_ece": comp_base["over_2_5_ece"],
                    "statsmodels_calibrated_over_2_5_ece": comp_candidate["over_2_5_ece"],
                    "incumbent_calibrated_btts_ece": comp_base["btts_ece"],
                    "statsmodels_calibrated_btts_ece": comp_candidate["btts_ece"],
                }
            )

        rows.append(
            {
                "fold": int(fold),
                "oos_start_utc": str(oos["kickoff_utc"].min()),
                "oos_end_utc": str(oos["kickoff_utc"].max()),
                "prediction_cutoff_utc": str(prediction_cutoff),
                "availability_source": availability_name,
                "predictor_pit_authority": "feature_source_max_available_at_utc",
                "model_fit_rows": int(len(model_fit)),
                "calibration_rows": int(len(calibration_common)),
                "training_rows": int(len(train)),
                "oos_rows": int(len(oos)),
                "common_evaluable_rows": int(len(common_oos)),
                "unsupported_rows": int(len(excluded_oos)),
                "common_coverage": float(len(common_oos) / len(oos)),
                "incumbent_score_logloss": raw_base["score_logloss"],
                "statsmodels_score_logloss": raw_candidate["score_logloss"],
                "incumbent_calibrated_score_logloss": cal_base["score_logloss"],
                "statsmodels_calibrated_score_logloss": cal_candidate["score_logloss"],
                "incumbent_score_top1": raw_base["score_top1"],
                "statsmodels_score_top1": raw_candidate["score_top1"],
                "incumbent_calibrated_score_top1": cal_base["score_top1"],
                "statsmodels_calibrated_score_top1": cal_candidate["score_top1"],
                "incumbent_score_top3": raw_base["score_top3"],
                "statsmodels_score_top3": raw_candidate["score_top3"],
                "incumbent_calibrated_score_top3": cal_base["score_top3"],
                "statsmodels_calibrated_score_top3": cal_candidate["score_top3"],
                "incumbent_score_top4": raw_base["score_top4"],
                "statsmodels_score_top4": raw_candidate["score_top4"],
                "incumbent_calibrated_score_top4": cal_base["score_top4"],
                "statsmodels_calibrated_score_top4": cal_candidate["score_top4"],
                "incumbent_over_2_5_logloss": raw_base["over_2_5_logloss"],
                "statsmodels_over_2_5_logloss": raw_candidate["over_2_5_logloss"],
                "incumbent_calibrated_over_2_5_logloss": cal_base["over_2_5_logloss"],
                "statsmodels_calibrated_over_2_5_logloss": cal_candidate["over_2_5_logloss"],
                "incumbent_over_2_5_brier": raw_base["over_2_5_brier"],
                "statsmodels_over_2_5_brier": raw_candidate["over_2_5_brier"],
                "incumbent_calibrated_over_2_5_brier": cal_base["over_2_5_brier"],
                "statsmodels_calibrated_over_2_5_brier": cal_candidate["over_2_5_brier"],
                "incumbent_over_2_5_ece": raw_base["over_2_5_ece"],
                "statsmodels_over_2_5_ece": raw_candidate["over_2_5_ece"],
                "incumbent_calibrated_over_2_5_ece": cal_base["over_2_5_ece"],
                "statsmodels_calibrated_over_2_5_ece": cal_candidate["over_2_5_ece"],
                "incumbent_btts_logloss": raw_base["btts_logloss"],
                "statsmodels_btts_logloss": raw_candidate["btts_logloss"],
                "incumbent_calibrated_btts_logloss": cal_base["btts_logloss"],
                "statsmodels_calibrated_btts_logloss": cal_candidate["btts_logloss"],
                "incumbent_btts_brier": raw_base["btts_brier"],
                "statsmodels_btts_brier": raw_candidate["btts_brier"],
                "incumbent_calibrated_btts_brier": cal_base["btts_brier"],
                "statsmodels_calibrated_btts_brier": cal_candidate["btts_brier"],
                "incumbent_btts_ece": raw_base["btts_ece"],
                "statsmodels_btts_ece": raw_candidate["btts_ece"],
                "incumbent_calibrated_btts_ece": cal_base["btts_ece"],
                "statsmodels_calibrated_btts_ece": cal_candidate["btts_ece"],
                "incumbent_calibration_temperature": incumbent_temperature,
                "statsmodels_calibration_temperature": challenger_temperature,
                "raw_calibration_logloss_incumbent": float(
                    incumbent_temperature_info["raw_calibration_logloss"]
                ),
                "calibrated_calibration_logloss_incumbent": float(
                    incumbent_temperature_info["calibrated_calibration_logloss"]
                ),
                "raw_calibration_logloss_statsmodels": float(
                    challenger_temperature_info["raw_calibration_logloss"]
                ),
                "calibrated_calibration_logloss_statsmodels": float(
                    challenger_temperature_info["calibrated_calibration_logloss"]
                ),
                "competition_slices": competition_slices,
                "selection_performed": False,
                "frozen_holdout_used": False,
                "production_usable": False,
                "pit_training_boundary_valid": bool(
                    (train["kickoff_utc"] < prediction_cutoff).all()
                    and (train["source_available_at_utc"] <= prediction_cutoff).all()
                ),
                "pit_oos_case_valid": bool(oos_pit_valid.all()),
                "same_kickoff_split_avoided": bool(
                    train["kickoff_utc"].max() < oos["kickoff_utc"].min()
                ),
                "calibration_precedes_oos": bool(
                    calibration["kickoff_utc"].max() < oos["kickoff_utc"].min()
                ),
            }
        )

        fold += 1
        start = end

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("statsmodels OOS produced no evaluation folds")

    experiment_config = {
        "research_id": "statsmodels_poisson_score_v2",
        "adapter": "src/research/external_oss/statsmodels_poisson.py",
        "dependency_pin": "statsmodels==0.15.0",
        "regularization_alpha": 0.1,
        "regularization": "ridge_L2",
        "min_train": int(min_train),
        "oos_block": int(oos_block),
        "cutoff_buffer_minutes": int(cutoff_buffer_minutes),
        "calibration_fraction": float(calibration_fraction),
        "calibration_min_rows": int(calibration_min_rows),
        "calibration_method": "joint_score_temperature_scaling",
        "selection_performed": False,
        "frozen_holdout_used": False,
        "production_usable": False,
        "target_contract": "Score",
        "feature_set_id": "statsmodels_score_formula_home_away_competition_v1",
        "predictor_pit_authority": "feature_source_max_available_at_utc",
    }

    return {
        "schema_version": 2,
        "status": "RESEARCH_OOS_READY",
        "research_only": True,
        "production_usable": False,
        "selection_performed": False,
        "frozen_holdout_used": False,
        "handoff": handoff,
        "experiment_config": experiment_config,
        "experiment_config_fingerprint": _experiment_fingerprint(experiment_config),
        "fold_count": int(len(result)),
        "rows": result.to_dict(orient="records"),
    }


def run_from_environment(history: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
    """Environment-backed wrapper used by GitHub Actions."""
    tests_passed = os.getenv("TESTS_PASSED", "false").lower() == "true"
    audit_passed = os.getenv("AUDIT_PASSED", "false").lower() == "true"
    return run_statsmodels_score_oos(
        history,
        tests_passed=tests_passed,
        audit_passed=audit_passed,
        **kwargs,
    )
