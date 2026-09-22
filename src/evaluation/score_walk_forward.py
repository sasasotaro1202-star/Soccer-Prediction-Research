from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.models.dixon_coles import fit_dixon_coles_model, predict_dixon_coles_distribution
from src.prediction.secondary_outputs import (
    fit_recency_score_rate_model,
    fit_score_rate_model,
    predict_score_distribution,
)


def _binary_logloss(y: int, p: float) -> float:
    p = float(np.clip(p, 1e-9, 1 - 1e-9))
    return float(-(y * math.log(p) + (1 - y) * math.log(1 - p)))


def _score_block_metrics(
    block: pd.DataFrame,
    model: dict,
    *,
    distribution_fn=predict_score_distribution,
) -> dict[str, float]:
    exact_hits = top3_hits = top4_hits = 0
    exact_losses = []
    home_abs = away_abs = total_abs = 0.0
    over25_logloss = []
    over25_brier = []
    btts_logloss = []
    btts_brier = []

    for row in block.itertuples(index=False):
        actual_h = int(row.home_goals)
        actual_a = int(row.away_goals)
        dist = distribution_fn(
            model,
            row.home_team,
            row.away_team,
            row.competition if hasattr(row, "competition") else None,
            max_goals=12,
            neutral_venue=getattr(row, "neutral_venue", None),
        )
        lookup = {(int(h), int(a)): float(p) for h, a, p in dist}
        actual_prob = lookup.get((actual_h, actual_a), 0.0)
        exact_losses.append(-math.log(max(actual_prob, 1e-12)))

        ranked = sorted(dist, key=lambda x: (-float(x[2]), int(x[0]), int(x[1])))
        top3 = {(int(h), int(a)) for h, a, _ in ranked[:3]}
        top4 = {(int(h), int(a)) for h, a, _ in ranked[:4]}
        exact_hits += int((actual_h, actual_a) == (int(ranked[0][0]), int(ranked[0][1])))
        top3_hits += int((actual_h, actual_a) in top3)
        top4_hits += int((actual_h, actual_a) in top4)

        best_h, best_a = int(ranked[0][0]), int(ranked[0][1])
        home_abs += abs(best_h - actual_h)
        away_abs += abs(best_a - actual_a)
        total_abs += abs((best_h + best_a) - (actual_h + actual_a))

        probs = np.asarray([float(p) for _, _, p in dist], dtype=float)
        totals = np.asarray([int(h + a) for h, a, _ in dist], dtype=int)
        btts_mask = np.asarray([int(h) >= 1 and int(a) >= 1 for h, a, _ in dist], dtype=bool)
        p_over25 = float(probs[totals >= 3].sum())
        p_btts = float(probs[btts_mask].sum())
        y_over25 = int(actual_h + actual_a >= 3)
        y_btts = int(actual_h >= 1 and actual_a >= 1)
        over25_logloss.append(_binary_logloss(y_over25, p_over25))
        over25_brier.append((p_over25 - y_over25) ** 2)
        btts_logloss.append(_binary_logloss(y_btts, p_btts))
        btts_brier.append((p_btts - y_btts) ** 2)

    n = max(1, len(block))
    return {
        "n": float(len(block)),
        "score_logloss": float(np.mean(exact_losses)),
        "exact_score_hit_rate": float(exact_hits / n),
        "top3_score_hit_rate": float(top3_hits / n),
        "top4_score_hit_rate": float(top4_hits / n),
        "home_goals_mae": float(home_abs / n),
        "away_goals_mae": float(away_abs / n),
        "total_goals_mae": float(total_abs / n),
        "over_2_5_logloss": float(np.mean(over25_logloss)),
        "over_2_5_brier": float(np.mean(over25_brier)),
        "btts_logloss": float(np.mean(btts_logloss)),
        "btts_brier": float(np.mean(btts_brier)),
    }


def run_score_walk_forward(
    df: pd.DataFrame,
    *,
    min_train: int = 1000,
    oos_block: int = 2000,
) -> pd.DataFrame:
    required = {
        "kickoff_utc",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "pit_verified",
        "source_available_at_utc",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Score OOS data missing columns: {missing}")

    d = df.copy()
    d["kickoff_utc"] = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    d["home_goals"] = pd.to_numeric(d["home_goals"], errors="coerce")
    d["away_goals"] = pd.to_numeric(d["away_goals"], errors="coerce")
    d["source_available_at_utc"] = pd.to_datetime(
        d["source_available_at_utc"], utc=True, errors="coerce"
    )
    d["pit_verified"] = d["pit_verified"].astype("boolean")
    # PIT applies to prediction inputs. The OOS target/result label is evaluated
    # after the fact and does not need its publication timestamp to be known at the
    # prediction cutoff. Training rows remain strictly constrained by
    # source_available_at_utc <= prediction_cutoff below.
    d = d[
        d["pit_verified"].eq(True)
        & d["kickoff_utc"].notna()
        & d["home_goals"].notna()
        & d["away_goals"].notna()
    ].sort_values("kickoff_utc", kind="mergesort").reset_index(drop=True)

    if len(d) < min_train + oos_block:
        raise ValueError(
            f"Not enough PIT-verified score rows: {len(d)}; need at least {min_train + oos_block}"
        )

    rows = []
    start = int(min_train)
    while start < len(d):
        end = min(start + int(oos_block), len(d))
        train = d.iloc[:start]
        oos = d.iloc[start:end]
        prediction_cutoff = oos["kickoff_utc"].min() - pd.Timedelta(minutes=60)
        train = train[train["source_available_at_utc"] <= prediction_cutoff].copy()
        if len(train) < int(min_train):
            raise ValueError(
                "Not enough PIT-available score training rows before OOS cutoff: "
                f"{len(train)}; need at least {min_train}"
            )
        model = fit_score_rate_model(train)
        metrics = _score_block_metrics(oos, model)

        # Challenger family: time-decay half-life is tuned on development OOS only.
        # Each half-life is evaluated independently; no locked OOS row is used here.
        for half_life_days in (180.0, 365.0, 730.0, 1095.0):
            tag = int(half_life_days)
            try:
                recency_model = fit_recency_score_rate_model(
                    train,
                    half_life_days=half_life_days,
                )
                recency_metrics = _score_block_metrics(oos, recency_model)
                metrics.update({
                    f"recency_d{tag}_{k}": v
                    for k, v in recency_metrics.items()
                    if k != "n"
                })
                metrics[f"recency_d{tag}_status"] = "PASS"
                metrics[f"recency_d{tag}_error"] = ""
            except Exception as exc:
                for metric in (
                    "score_logloss",
                    "exact_score_hit_rate",
                    "top3_score_hit_rate",
                    "top4_score_hit_rate",
                    "home_goals_mae",
                    "away_goals_mae",
                    "total_goals_mae",
                    "over_2_5_logloss",
                    "over_2_5_brier",
                    "btts_logloss",
                    "btts_brier",
                ):
                    metrics[f"recency_d{tag}_{metric}"] = float("nan")
                metrics[f"recency_d{tag}_status"] = "ERROR"
                metrics[f"recency_d{tag}_error"] = f"{type(exc).__name__}: {exc}"

        # Challenger: Dixon-Coles low-score dependence correction. A challenger
        # error is recorded, not allowed to contaminate primary Poisson metrics.
        try:
            dc_model = fit_dixon_coles_model(train)
            dc_metrics = _score_block_metrics(
                oos,
                dc_model,
                distribution_fn=predict_dixon_coles_distribution,
            )
            metrics.update({f"dc_{k}": v for k, v in dc_metrics.items() if k != "n"})
            metrics["dc_rho"] = float(dc_model.get("rho", 0.0))
            metrics["dc_rho_fit_used"] = bool(dc_model.get("rho_fit_used", False))
            metrics["dc_status"] = "PASS"
            metrics["dc_error"] = ""
        except Exception as exc:
            metrics.update(
                {
                    "dc_score_logloss": float(metrics["score_logloss"]),
                    "dc_exact_score_hit_rate": float(metrics["exact_score_hit_rate"]),
                    "dc_top3_score_hit_rate": float(metrics["top3_score_hit_rate"]),
                    "dc_top4_score_hit_rate": float(metrics["top4_score_hit_rate"]),
                    "dc_home_goals_mae": float(metrics["home_goals_mae"]),
                    "dc_away_goals_mae": float(metrics["away_goals_mae"]),
                    "dc_total_goals_mae": float(metrics["total_goals_mae"]),
                    "dc_over_2_5_logloss": float(metrics["over_2_5_logloss"]),
                    "dc_over_2_5_brier": float(metrics["over_2_5_brier"]),
                    "dc_btts_logloss": float(metrics["btts_logloss"]),
                    "dc_btts_brier": float(metrics["btts_brier"]),
                    "dc_rho": 0.0,
                    "dc_rho_fit_used": False,
                    "dc_status": "ERROR",
                    "dc_error": f"{type(exc).__name__}: {exc}",
                }
            )
        metrics.update(
            {
                "oos_start": str(oos["kickoff_utc"].min()),
                "oos_end": str(oos["kickoff_utc"].max()),
                "score_training_cutoff": str(prediction_cutoff),
                "score_training_rows": int(len(train)),
                "competitions": "|".join(sorted(oos["competition"].astype(str).unique()))
                if "competition" in oos.columns
                else "",
            }
        )
        rows.append(metrics)
        start = end

    return pd.DataFrame(rows)
