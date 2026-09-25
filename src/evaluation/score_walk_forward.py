from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.models.dixon_coles import fit_dixon_coles_model, predict_dixon_coles_distribution
from src.models.negative_binomial import (
    fit_negative_binomial_score_model,
    predict_negative_binomial_distribution,
)
from src.prediction.secondary_outputs import (
    fit_recency_score_rate_model,
    fit_time_decay_score_rate_model,
    fit_neutral_aware_score_rate_model,
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
    risk_scores = []
    risk_exact_losses = []
    risk_top3_hits = []
    risk_over25_logloss = []
    risk_over25_brier = []
    risk_btts_logloss = []
    risk_btts_brier = []

    for row in block.itertuples(index=False):
        actual_h = int(row.home_goals)
        actual_a = int(row.away_goals)
        neutral_venue = bool(
            getattr(row, "neutral_venue", False)
        ) if hasattr(row, "neutral_venue") else False
        if isinstance(neutral_venue, float) and np.isnan(neutral_venue):
            neutral_venue = False
        distribution_kwargs = {"max_goals": 12}
        if str(model.get("method", "")).startswith("neutral_aware_"):
            distribution_kwargs["neutral_venue"] = neutral_venue
        dist = distribution_fn(
            model,
            row.home_team,
            row.away_team,
            row.competition if hasattr(row, "competition") else None,
            **distribution_kwargs,
        )
        lookup = {(int(h), int(a)): float(p) for h, a, p in dist}
        actual_prob = lookup.get((actual_h, actual_a), 0.0)
        exact_losses.append(-math.log(max(actual_prob, 1e-12)))

        ranked = sorted(dist, key=lambda x: (-float(x[2]), int(x[0]), int(x[1])))
        top_probs = [float(x[2]) for x in ranked[:2]]
        entropy = -sum(
            float(np.clip(p, 1e-12, 1.0)) * math.log(float(np.clip(p, 1e-12, 1.0)))
            for _, _, p in dist
        ) / max(math.log(len(dist)), 1e-12)
        margin_risk = 1.0 - float(np.clip((top_probs[0] - top_probs[1]) if len(top_probs) >= 2 else top_probs[0], 0.0, 1.0))
        score_risk = float(np.clip(0.60 * entropy + 0.40 * margin_risk, 0.0, 1.0))
        risk_scores.append(score_risk)
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
        if score_risk >= 0.66:
            risk_exact_losses.append(-math.log(max(actual_prob, 1e-12)))
            risk_top3_hits.append(int((actual_h, actual_a) in top3))
            risk_over25_logloss.append(over25_logloss[-1])
            risk_over25_brier.append(over25_brier[-1])
            risk_btts_logloss.append(btts_logloss[-1])
            risk_btts_brier.append(btts_brier[-1])

    n = max(1, len(block))
    high_n = len(risk_exact_losses)
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
        "mean_score_risk": float(np.mean(risk_scores)) if risk_scores else float("nan"),
        "high_risk_score_n": float(high_n),
        "high_risk_score_share": float(high_n / n),
        "high_risk_score_logloss": float(np.mean(risk_exact_losses)) if risk_exact_losses else float("nan"),
        "high_risk_top3_score_hit_rate": float(np.mean(risk_top3_hits)) if risk_top3_hits else float("nan"),
        "high_risk_over_2_5_logloss": float(np.mean(risk_over25_logloss)) if risk_over25_logloss else float("nan"),
        "high_risk_over_2_5_brier": float(np.mean(risk_over25_brier)) if risk_over25_brier else float("nan"),
        "high_risk_btts_logloss": float(np.mean(risk_btts_logloss)) if risk_btts_logloss else float("nan"),
        "high_risk_btts_brier": float(np.mean(risk_btts_brier)) if risk_btts_brier else float("nan"),
    }


def _advance_past_same_kickoff(df: pd.DataFrame, index: int) -> int:
    """Move a row-count boundary past all rows sharing the same kickoff time."""
    boundary = int(index)
    if boundary <= 0 or boundary >= len(df):
        return boundary
    kickoff = df.iloc[boundary - 1]["kickoff_utc"]
    while boundary < len(df) and df.iloc[boundary]["kickoff_utc"] == kickoff:
        boundary += 1
    return boundary


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
    d = d[
        d["pit_verified"].eq(True)
        & d["kickoff_utc"].notna()
        & d["home_goals"].notna()
        & d["away_goals"].notna()
        & d["source_available_at_utc"].notna()
    ].sort_values("kickoff_utc", kind="mergesort").reset_index(drop=True)

    if len(d) < min_train + oos_block:
        raise ValueError(
            f"Not enough PIT-verified score rows: {len(d)}; need at least {min_train + oos_block}"
        )

    rows = []
    start = _advance_past_same_kickoff(d, int(min_train))
    while start < len(d):
        end = _advance_past_same_kickoff(d, min(start + int(oos_block), len(d)))
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

        # Challenger: neutral-venue-aware rates. Standard fixtures match primary exactly;
        # neutral fixtures avoid importing a home-field asymmetry that was not observed.
        try:
            neutral_model = fit_neutral_aware_score_rate_model(train)
            neutral_metrics = _score_block_metrics(oos, neutral_model)
            metrics.update({f"neutral_aware_{k}": v for k, v in neutral_metrics.items() if k != "n"})
            metrics["neutral_aware_status"] = "PASS"
            metrics["neutral_aware_error"] = ""
        except Exception as exc:
            for key in (
                "score_logloss", "exact_score_hit_rate", "top3_score_hit_rate",
                "top4_score_hit_rate", "home_goals_mae", "away_goals_mae",
                "total_goals_mae", "over_2_5_logloss", "over_2_5_brier",
                "btts_logloss", "btts_brier",
            ):
                metrics[f"neutral_aware_{key}"] = float("nan")
            metrics["neutral_aware_status"] = "ERROR"
            metrics["neutral_aware_error"] = f"{type(exc).__name__}: {exc}"

        # Challenger: recency-weighted venue/team rates. Never alters primary metrics.
        try:
            recency_model = fit_recency_score_rate_model(train)
            recency_metrics = _score_block_metrics(oos, recency_model)
            metrics.update({f"recency_{k}": v for k, v in recency_metrics.items() if k != "n"})
            metrics["recency_status"] = "PASS"
            metrics["recency_error"] = ""
        except Exception as exc:
            metrics.update({
                "recency_score_logloss": float("nan"),
                "recency_exact_score_hit_rate": float("nan"),
                "recency_top3_score_hit_rate": float("nan"),
                "recency_top4_score_hit_rate": float("nan"),
                "recency_home_goals_mae": float("nan"),
                "recency_away_goals_mae": float("nan"),
                "recency_total_goals_mae": float("nan"),
                "recency_over_2_5_logloss": float("nan"),
                "recency_over_2_5_brier": float("nan"),
                "recency_btts_logloss": float("nan"),
                "recency_btts_brier": float("nan"),
                "recency_status": "ERROR",
                "recency_error": f"{type(exc).__name__}: {exc}",
            })

        # Challenger: elapsed-time-decay venue/team rates.
        try:
            time_decay_model = fit_time_decay_score_rate_model(train)
            time_decay_metrics = _score_block_metrics(oos, time_decay_model)
            metrics.update({f"time_decay_{k}": v for k, v in time_decay_metrics.items() if k != "n"})
            metrics["time_decay_status"] = "PASS"
            metrics["time_decay_error"] = ""
        except Exception as exc:
            metrics.update({
                "time_decay_score_logloss": float("nan"),
                "time_decay_exact_score_hit_rate": float("nan"),
                "time_decay_top3_score_hit_rate": float("nan"),
                "time_decay_top4_score_hit_rate": float("nan"),
                "time_decay_home_goals_mae": float("nan"),
                "time_decay_away_goals_mae": float("nan"),
                "time_decay_total_goals_mae": float("nan"),
                "time_decay_over_2_5_logloss": float("nan"),
                "time_decay_over_2_5_brier": float("nan"),
                "time_decay_btts_logloss": float("nan"),
                "time_decay_btts_brier": float("nan"),
                "time_decay_status": "ERROR",
                "time_decay_error": f"{type(exc).__name__}: {exc}",
            })

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
        # Challenger: Negative-Binomial goal-count distribution for overdispersion.
        try:
            nb_model = fit_negative_binomial_score_model(train)
            nb_metrics = _score_block_metrics(
                oos,
                nb_model,
                distribution_fn=predict_negative_binomial_distribution,
            )
            metrics.update({f"negative_binomial_{k}": v for k, v in nb_metrics.items() if k != "n"})
            metrics["negative_binomial_home_dispersion"] = float(nb_model.get("home_dispersion", 0.0))
            metrics["negative_binomial_away_dispersion"] = float(nb_model.get("away_dispersion", 0.0))
            metrics["negative_binomial_status"] = "PASS"
            metrics["negative_binomial_error"] = ""
        except Exception as exc:
            for key in (
                "score_logloss", "exact_score_hit_rate", "top3_score_hit_rate",
                "top4_score_hit_rate", "home_goals_mae", "away_goals_mae",
                "total_goals_mae", "over_2_5_logloss", "over_2_5_brier",
                "btts_logloss", "btts_brier",
            ):
                metrics[f"negative_binomial_{key}"] = float("nan")
            metrics["negative_binomial_home_dispersion"] = float("nan")
            metrics["negative_binomial_away_dispersion"] = float("nan")
            metrics["negative_binomial_status"] = "ERROR"
            metrics["negative_binomial_error"] = f"{type(exc).__name__}: {exc}"

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
