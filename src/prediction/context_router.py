from __future__ import annotations

import numpy as np
import pandas as pd


def routing_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the exact prediction-time context used by research routing."""
    d = frame.copy()

    if "elo_diff" in d.columns:
        elo = pd.to_numeric(d["elo_diff"], errors="coerce")
        d["routing_strength_gap"] = pd.cut(
            elo,
            bins=[-np.inf, -200.0, -75.0, 75.0, 200.0, np.inf],
            labels=["LARGE_AWAY", "AWAY", "EVEN", "HOME", "LARGE_HOME"],
        ).astype("string").fillna("MISSING")
    else:
        d["routing_strength_gap"] = "MISSING"

    home_goal = pd.to_numeric(
        d["home_goal_total_avg_5"] if "home_goal_total_avg_5" in d.columns else pd.Series(np.nan, index=d.index),
        errors="coerce",
    )
    away_goal = pd.to_numeric(
        d["away_goal_total_avg_5"] if "away_goal_total_avg_5" in d.columns else pd.Series(np.nan, index=d.index),
        errors="coerce",
    )
    goal_env = (home_goal + away_goal) / 2.0
    d["routing_scoring_environment"] = pd.cut(
        goal_env,
        bins=[-np.inf, 1.8, 2.3, 2.8, np.inf],
        labels=["LOW", "MID_LOW", "MID_HIGH", "HIGH"],
    ).astype("string").fillna("MISSING")

    rest_source = (
        d["rest_diff_hours"]
        if "rest_diff_hours" in d.columns
        else pd.Series(np.nan, index=d.index, dtype=float)
    )
    rest = pd.to_numeric(rest_source, errors="coerce")
    d["routing_rest"] = pd.cut(
        rest,
        bins=[-np.inf, -24.0, -6.0, 6.0, 24.0, np.inf],
        labels=["AWAY_MAJOR", "AWAY_SMALL", "EVEN", "HOME_SMALL", "HOME_MAJOR"],
    ).astype("string").fillna("MISSING")

    if "neutral_venue_known" in d.columns:
        known = d["neutral_venue_known"].astype("boolean")
        neutral = d.get("neutral_venue", pd.Series(False, index=d.index)).astype("boolean")
        d["routing_venue"] = np.where(
            known.fillna(False),
            np.where(neutral.fillna(False), "NEUTRAL", "HOME_AWAY"),
            "UNKNOWN",
        )
    else:
        d["routing_venue"] = "UNKNOWN"

    d["routing_context"] = (
        d.get("competition", pd.Series("__MISSING__", index=d.index)).astype("string").fillna("__MISSING__")
        + "|"
        + d["routing_strength_gap"].fillna("MISSING")
        + "|"
        + d["routing_scoring_environment"].fillna("MISSING")
        + "|"
        + d["routing_rest"].fillna("MISSING")
        + "|"
        + d["routing_venue"].astype("string")
    )
    return d


def context_route_candidates(row: pd.Series) -> list[tuple[str, str]]:
    """Return route keys from most specific to broadest."""
    routed = routing_context(pd.DataFrame([row])).iloc[0] if "routing_context" not in row.index else row
    full = str(routed.get("routing_context", ""))
    comp = str(routed.get("competition", "__MISSING__"))
    strength = str(routed.get("routing_strength_gap", "MISSING"))
    return [
        ("FULL", full),
        ("COMP_STRENGTH", f"{comp}|{strength}"),
        ("COMP", comp),
    ]


def lookup_context_weights(
    row: pd.Series,
    context_weights: dict[str, dict[str, float]] | None,
    fallback: dict[str, float],
) -> tuple[dict[str, float], str]:
    """Resolve learned production routing with hierarchical fallback."""
    if not context_weights:
        return dict(fallback), "GLOBAL"
    for level, key in context_route_candidates(row):
        learned = context_weights.get(f"{level}:{key}")
        if learned is not None:
            return {str(k): float(v) for k, v in learned.items()}, f"{level}:{key}"
    return dict(fallback), "GLOBAL"


def temperature_transform(proba: np.ndarray, temperature: float) -> np.ndarray:
    p = np.clip(np.asarray(proba, dtype=float), 1e-9, 1.0)
    logits = np.log(p) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    return out / out.sum(axis=1, keepdims=True)


def apply_contextual_temperatures(
    proba: np.ndarray,
    rows: pd.DataFrame,
    contextual_temperatures: dict[str, float] | None,
    fallback: float,
) -> np.ndarray:
    p = np.asarray(proba, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(rows) != len(p):
        raise ValueError("Probability/row shape mismatch in contextual temperature calibration")
    if not contextual_temperatures:
        return temperature_transform(p, float(fallback))
    routed = routing_context(rows)
    out = np.empty_like(p, dtype=float)
    for i, (_, row) in enumerate(routed.iterrows()):
        route = str(row.get("routing_context", ""))
        t = float(contextual_temperatures.get(route, fallback))
        if not np.isfinite(t) or t <= 0:
            t = float(fallback)
        out[i] = temperature_transform(p[i:i+1], t)[0]
    if not np.isfinite(out).all():
        raise ValueError("Contextual temperature calibration produced invalid probabilities")
    return out
