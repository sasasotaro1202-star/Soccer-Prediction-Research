from __future__ import annotations

"""Research-only last-resort forecast routing.

The fallback is deliberately weaker than an adopted production model, but it
must remain deterministic, PIT-aware and total: every structurally valid future
fixture receives a probability distribution rather than being dropped only
because a production registry is unavailable or a team is unseen.

It never promotes itself to production. Its purpose is continuity, coverage and
case-level research telemetry while stronger validated models are unavailable.
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_HISTORY_COLUMNS = {
    "kickoff_utc",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "pit_verified",
}

DEFAULT_HOME_GOALS = 1.25
DEFAULT_AWAY_GOALS = 1.08
SHRINKAGE = 20.0
HALF_LIFE_DAYS = 365.0
MAX_GOALS = 12
NEUTRAL_COMPETITIONS = {"AG_M", "AG_W"}
ELO_K = 20.0
ELO_HOME_ADV = 55.0
ELO_SHRINK_WEIGHT_CAP = 0.30


def _safe_float(value: Any, default: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _poisson_distribution(home_lambda: float, away_lambda: float, max_goals: int = MAX_GOALS) -> list[tuple[int, int, float]]:
    if max_goals < 1:
        raise ValueError("max_goals must be at least 1")
    home_lambda = float(np.clip(home_lambda, 0.05, 6.0))
    away_lambda = float(np.clip(away_lambda, 0.05, 6.0))
    rows: list[tuple[int, int, float]] = []
    for home_goals in range(max_goals + 1):
        ph = math.exp(-home_lambda) * home_lambda**home_goals / math.factorial(home_goals)
        for away_goals in range(max_goals + 1):
            pa = math.exp(-away_lambda) * away_lambda**away_goals / math.factorial(away_goals)
            rows.append((home_goals, away_goals, ph * pa))
    total = sum(p for _, _, p in rows)
    total = total if total > 0 and np.isfinite(total) else 1.0
    return [(h, a, float(p / total)) for h, a, p in rows]


def _load_pit_history(history_path: str, prediction_time: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(history_path)
    meta: dict[str, Any] = {
        "history_path": str(path),
        "status": "UNAVAILABLE",
        "rows_loaded": 0,
        "rows_pit_eligible": 0,
        "pit_source_time_enforced": False,
        "availability_source_counts": {
            "source_available_at_utc": 0,
            "versioned_result_available_at_utc": 0,
        },
        "versioned_result_verified_rows": 0,
    }
    if not path.is_file() or path.stat().st_size <= 0:
        meta["reason"] = "history_file_missing_or_empty"
        return pd.DataFrame(), meta

    # The normalized history is a repository artifact. Read only columns needed
    # for the fallback so frequent matchday refreshes remain cheap.
    frame = pd.read_csv(path)
    meta["rows_loaded"] = int(len(frame))
    missing = sorted(REQUIRED_HISTORY_COLUMNS - set(frame.columns))
    if missing:
        meta["reason"] = f"missing_history_columns:{missing}"
        return pd.DataFrame(), meta

    d = frame.copy()
    d["kickoff_utc"] = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    d["home_goals"] = pd.to_numeric(d["home_goals"], errors="coerce")
    d["away_goals"] = pd.to_numeric(d["away_goals"], errors="coerce")
    d["pit_verified"] = d["pit_verified"].astype("string").str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )
    versioned_verified = (
        d["versioned_result_evidence_status"].astype("string").str.strip().str.upper().eq("VERIFIED")
        if "versioned_result_evidence_status" in d.columns
        else pd.Series(False, index=d.index)
    )
    d["fallback_result_pit_verified"] = d["pit_verified"] | versioned_verified
    d = d.dropna(subset=["kickoff_utc", "home_goals", "away_goals", "home_team", "away_team"])
    d = d[d["fallback_result_pit_verified"]].copy()
    d = d[d["kickoff_utc"] < prediction_time].copy()
    meta["versioned_result_verified_rows"] = int(versioned_verified.sum())

    # Prefer explicit source availability. For this fallback, the only
    # historical signal consumed is completed match outcome, so a versioned
    # result-publication timestamp is also valid provenance when the richer
    # feature-source timestamp is absent.
    source_available = (
        pd.to_datetime(d["source_available_at_utc"], utc=True, errors="coerce")
        if "source_available_at_utc" in d.columns
        else pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns, UTC]")
    )
    versioned_available = (
        pd.to_datetime(d["versioned_result_available_at_utc"], utc=True, errors="coerce")
        if "versioned_result_available_at_utc" in d.columns
        else pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns, UTC]")
    )
    explicit_available = source_available.where(source_available.notna(), versioned_available)
    eligible = explicit_available.notna() & (explicit_available <= prediction_time)
    d["fallback_source_available_at_utc"] = explicit_available
    d = d[eligible].copy()
    meta["pit_source_time_enforced"] = True
    meta["availability_source_counts"] = {
        "source_available_at_utc": int((source_available.notna() & eligible).sum()),
        "versioned_result_available_at_utc": int(
            (source_available.isna() & versioned_available.notna() & eligible).sum()
        ),
    }
    if d.empty:
        meta["reason"] = "no_explicit_pit_availability_before_cutoff"
        return pd.DataFrame(), meta

    d["home_team"] = d["home_team"].astype(str).str.strip()
    d["away_team"] = d["away_team"].astype(str).str.strip()
    d["competition"] = d.get("competition", pd.Series("UNKNOWN", index=d.index)).astype(str).str.strip().str.upper()
    d = d[
        d["home_team"].ne("")
        & d["away_team"].ne("")
        & d["home_goals"].ge(0)
        & d["away_goals"].ge(0)
    ].copy()
    d = d.sort_values(["kickoff_utc", "match_id" if "match_id" in d.columns else "home_team"], kind="mergesort")
    meta["rows_pit_eligible"] = int(len(d))
    meta["status"] = "READY" if not d.empty else "NO_PIT_HISTORY"
    return d.reset_index(drop=True), meta


def _weighted_means(frame: pd.DataFrame, prediction_time: pd.Timestamp) -> tuple[float, float]:
    if frame.empty:
        return DEFAULT_HOME_GOALS, DEFAULT_AWAY_GOALS
    age_days = (
        prediction_time - pd.to_datetime(frame["kickoff_utc"], utc=True)
    ).dt.total_seconds() / 86400.0
    weights = np.exp(-np.log(2.0) * np.maximum(age_days, 0.0) / HALF_LIFE_DAYS)
    weights = np.asarray(weights, dtype=float)
    weights = np.maximum(weights, 1e-12)
    home_mean = float(np.average(frame["home_goals"].to_numpy(float), weights=weights))
    away_mean = float(np.average(frame["away_goals"].to_numpy(float), weights=weights))
    return home_mean, away_mean


def _team_rates(frame: pd.DataFrame, prediction_time: pd.Timestamp) -> dict[str, dict[str, float]]:
    if frame.empty:
        return {}
    d = frame.copy()
    age_days = (
        prediction_time - pd.to_datetime(d["kickoff_utc"], utc=True)
    ).dt.total_seconds() / 86400.0
    d["_weight"] = np.exp(-np.log(2.0) * np.maximum(age_days, 0.0) / HALF_LIFE_DAYS)

    global_home, global_away = _weighted_means(frame, prediction_time)
    global_overall = max((global_home + global_away) / 2.0, 0.05)

    rows: dict[str, dict[str, float]] = {}
    for team, g in d.groupby("home_team", sort=False):
        weight = float(g["_weight"].sum())
        scored = float(np.sum(g["home_goals"] * g["_weight"]))
        conceded = float(np.sum(g["away_goals"] * g["_weight"]))
        rows.setdefault(str(team), {})
        rows[str(team)]["home_scored"] = (scored + SHRINKAGE * global_home) / (weight + SHRINKAGE)
        rows[str(team)]["home_conceded"] = (conceded + SHRINKAGE * global_away) / (weight + SHRINKAGE)
        rows[str(team)]["home_weight"] = weight

    for team, g in d.groupby("away_team", sort=False):
        weight = float(g["_weight"].sum())
        scored = float(np.sum(g["away_goals"] * g["_weight"]))
        conceded = float(np.sum(g["home_goals"] * g["_weight"]))
        rows.setdefault(str(team), {})
        rows[str(team)]["away_scored"] = (scored + SHRINKAGE * global_away) / (weight + SHRINKAGE)
        rows[str(team)]["away_conceded"] = (conceded + SHRINKAGE * global_home) / (weight + SHRINKAGE)
        rows[str(team)]["away_weight"] = weight

    for team in rows:
        r = rows[team]
        total_weight = float(r.get("home_weight", 0.0) + r.get("away_weight", 0.0))
        scored_num = float(
            r.get("home_scored", global_home) * r.get("home_weight", 0.0)
            + r.get("away_scored", global_away) * r.get("away_weight", 0.0)
        )
        conceded_num = float(
            r.get("home_conceded", global_away) * r.get("home_weight", 0.0)
            + r.get("away_conceded", global_home) * r.get("away_weight", 0.0)
        )
        r["scored"] = (scored_num + SHRINKAGE * global_overall) / (total_weight + SHRINKAGE)
        r["conceded"] = (conceded_num + SHRINKAGE * global_overall) / (total_weight + SHRINKAGE)
        r["matches"] = total_weight
    return rows


def _elo_probabilities(
    frame: pd.DataFrame,
    home_team: str,
    away_team: str,
    competition: str,
    neutral_venue: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    ratings: dict[str, float] = {}
    for row in frame.sort_values(["kickoff_utc", "match_id" if "match_id" in frame.columns else "home_team"], kind="mergesort").itertuples(index=False):
        home = str(row.home_team)
        away = str(row.away_team)
        comp = str(row.competition)
        event_neutral = comp in NEUTRAL_COMPETITIONS
        he = float(ratings.get(home, 1500.0))
        ae = float(ratings.get(away, 1500.0))
        adv = 0.0 if event_neutral else ELO_HOME_ADV
        expected = 1.0 / (1.0 + 10.0 ** (-((he + adv) - ae) / 400.0))
        actual = 1.0 if float(row.home_goals) > float(row.away_goals) else (
            0.5 if float(row.home_goals) == float(row.away_goals) else 0.0
        )
        delta = ELO_K * (actual - expected)
        ratings[home] = he + delta
        ratings[away] = ae - delta

    home_rating = float(ratings.get(str(home_team), 1500.0))
    away_rating = float(ratings.get(str(away_team), 1500.0))
    adv = 0.0 if neutral_venue else ELO_HOME_ADV
    p_home = 1.0 / (1.0 + 10.0 ** (-((home_rating + adv) - away_rating) / 400.0))
    # ELO draw probability is constructed conservatively from proximity to
    # parity rather than pretending the binary ELO model estimates draws.
    proximity = float(np.exp(-abs((home_rating + adv) - away_rating) / 220.0))
    p_draw = float(np.clip(0.18 + 0.18 * proximity, 0.18, 0.36))
    remaining = max(1.0 - p_draw, 1e-12)
    p_home = float(np.clip(p_home, 0.02, 0.98))
    # Renormalize the win split after reserving a draw mass.
    p_home = float(np.clip(p_home * remaining, 0.01, remaining - 0.01))
    p_away = remaining - p_home
    probs = np.asarray([p_home, p_draw, p_away], dtype=float)
    probs /= probs.sum()
    support = {
        "home_rating": home_rating,
        "away_rating": away_rating,
        "home_rating_games": int(
            sum(1 for x in frame["home_team"].astype(str).tolist() if x == str(home_team))
            + sum(1 for x in frame["away_team"].astype(str).tolist() if x == str(home_team))
        ),
        "away_rating_games": int(
            sum(1 for x in frame["home_team"].astype(str).tolist() if x == str(away_team))
            + sum(1 for x in frame["away_team"].astype(str).tolist() if x == str(away_team))
        ),
    }
    return probs, support


def _competition_means(
    frame: pd.DataFrame,
    prediction_time: pd.Timestamp,
) -> dict[str, tuple[float, float, float]]:
    if frame.empty or "competition" not in frame.columns:
        return {}
    global_home, global_away = _weighted_means(frame, prediction_time)
    out: dict[str, tuple[float, float, float]] = {}
    for competition, g in frame.groupby("competition", sort=False):
        home, away = _weighted_means(g, prediction_time)
        n = float(len(g))
        shrunk_home = (home * n + global_home * SHRINKAGE) / (n + SHRINKAGE)
        shrunk_away = (away * n + global_away * SHRINKAGE) / (n + SHRINKAGE)
        out[str(competition)] = (shrunk_home, shrunk_away, n)
    return out


def _resolve_lambdas(
    frame: pd.DataFrame,
    team_rates: dict[str, dict[str, float]],
    comp_means: dict[str, tuple[float, float, float]],
    home_team: str,
    away_team: str,
    competition: str,
    neutral_venue: bool,
    prediction_time: pd.Timestamp,
) -> tuple[float, float, dict[str, Any]]:
    global_home, global_away = _weighted_means(frame, prediction_time)
    comp = comp_means.get(str(competition))
    if comp is not None:
        base_home, base_away = float(comp[0]), float(comp[1])
        comp_rows = float(comp[2])
    else:
        base_home, base_away = global_home, global_away
        comp_rows = 0.0

    neutral_mean = max((base_home + base_away) / 2.0, 0.05)
    home_info = team_rates.get(str(home_team), {})
    away_info = team_rates.get(str(away_team), {})

    if neutral_venue:
        home_attack = float(home_info.get("scored", neutral_mean))
        home_defense = float(home_info.get("conceded", neutral_mean))
        away_attack = float(away_info.get("scored", neutral_mean))
        away_defense = float(away_info.get("conceded", neutral_mean))
        home_lambda = neutral_mean * (home_attack / neutral_mean) * (away_defense / neutral_mean)
        away_lambda = neutral_mean * (away_attack / neutral_mean) * (home_defense / neutral_mean)
    else:
        home_attack = float(home_info.get("home_scored", base_home))
        home_defense = float(home_info.get("home_conceded", base_away))
        away_attack = float(away_info.get("away_scored", base_away))
        away_defense = float(away_info.get("away_conceded", base_home))
        home_lambda = base_home * (home_attack / max(base_home, 0.05)) * (away_defense / max(base_home, 0.05))
        away_lambda = base_away * (away_attack / max(base_away, 0.05)) * (home_defense / max(base_away, 0.05))

    evidence = {
        "competition_history_rows": int(comp_rows),
        "home_team_history_rows": int(round(
            home_info.get("matches", 0.0)
        )),
        "away_team_history_rows": int(round(
            away_info.get("matches", 0.0)
        )),
        "data_level": (
            "TEAM_AND_COMPETITION"
            if home_info and away_info and comp_rows > 0
            else "TEAM_WITH_GLOBAL_OR_COMPETITION_PRIOR"
            if home_info or away_info
            else "GLOBAL_OR_COMPETITION_PRIOR"
        ),
    }
    return float(np.clip(home_lambda, 0.05, 5.0)), float(np.clip(away_lambda, 0.05, 5.0)), evidence


def predict_fallback_fixture(
    *,
    home_team: str,
    away_team: str,
    competition: str,
    history: pd.DataFrame,
    prediction_time: pd.Timestamp,
    neutral_venue: bool = False,
    team_rates: dict[str, dict[str, float]] | None = None,
    comp_means: dict[str, tuple[float, float, float]] | None = None,
    market_probabilities: tuple[float, float, float] | None = None,
    elo_probabilities: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    team_rates = team_rates if team_rates is not None else _team_rates(history, prediction_time)
    comp_means = comp_means if comp_means is not None else _competition_means(history, prediction_time)
    home_lambda, away_lambda, evidence = _resolve_lambdas(
        history,
        team_rates,
        comp_means,
        str(home_team),
        str(away_team),
        str(competition),
        bool(neutral_venue),
        prediction_time,
    )

    distribution = _poisson_distribution(home_lambda, away_lambda)
    home_probability = sum(p for h, a, p in distribution if h > a)
    draw_probability = sum(p for h, a, p in distribution if h == a)
    away_probability = sum(p for h, a, p in distribution if h < a)
    one_x_two = np.asarray([home_probability, draw_probability, away_probability], dtype=float)
    one_x_two /= max(float(one_x_two.sum()), 1e-12)
    elo_used = False
    elo_meta: dict[str, Any] = {}
    if elo_probabilities is not None:
        elo = np.asarray(elo_probabilities, dtype=float)
        if (
            elo.shape == (3,)
            and np.isfinite(elo).all()
            and np.all((elo >= 0.0) & (elo <= 1.0))
            and float(elo.sum()) > 0.0
        ):
            elo /= float(elo.sum())
            support_weight = min(
                ELO_SHRINK_WEIGHT_CAP,
                0.05 + 0.01 * min(
                    20.0,
                    float(
                        (team_rates.get(str(home_team), {}).get("matches", 0.0))
                        + (team_rates.get(str(away_team), {}).get("matches", 0.0))
                    ),
                ),
            )
            one_x_two = (1.0 - support_weight) * one_x_two + support_weight * elo
            one_x_two /= max(float(one_x_two.sum()), 1e-12)
            elo_used = True
            elo_meta = {"elo_weight": float(support_weight)}

    market_used = False
    if market_probabilities is not None:
        market = np.asarray(market_probabilities, dtype=float)
        if (
            market.shape == (3,)
            and np.isfinite(market).all()
            and np.all((market >= 0.0) & (market <= 1.0))
            and float(market.sum()) > 0.0
        ):
            market /= float(market.sum())
            # Blend only the final 1X2 prior. Score/market totals remain
            # generated by the PIT-safe Poisson baseline, avoiding an invented
            # score distribution from a market quote.
            one_x_two = 0.65 * one_x_two + 0.35 * market
            one_x_two /= max(float(one_x_two.sum()), 1e-12)
            market_used = True

    ranked = sorted(distribution, key=lambda x: x[2], reverse=True)[:3]
    market_over_25 = sum(p for h, a, p in distribution if h + a > 2)
    market_under_25 = sum(p for h, a, p in distribution if h + a < 3)
    btts_yes = sum(p for h, a, p in distribution if h >= 1 and a >= 1)
    btts_no = sum(p for h, a, p in distribution if h == 0 or a == 0)

    labels = (str(home_team), "引き分け", str(away_team))
    label = labels[int(np.argmax(one_x_two))]
    return {
        "p_home": float(one_x_two[0]),
        "p_draw": float(one_x_two[1]),
        "p_away": float(one_x_two[2]),
        "result_prediction": label,
        "result_prediction_probability": float(one_x_two.max()),
        "scores": [
            {"score": f"{h}-{a}", "probability": float(p)}
            for h, a, p in ranked
        ],
        "over_2_5_probability": float(market_over_25),
        "under_2_5_probability": float(market_under_25),
        "btts_yes_probability": float(btts_yes),
        "btts_no_probability": float(btts_no),
        "home_lambda": home_lambda,
        "away_lambda": away_lambda,
        "evidence": {
            **evidence,
            **elo_meta,
            "elo_prior_used": elo_used,
            "market_prior_used": market_used,
            "market_weight": 0.35 if market_used else 0.0,
        },
    }


def build_fallback_forecast(
    fixtures: pd.DataFrame,
    prediction_time: pd.Timestamp,
    *,
    history_path: str = "artifacts/normalized_history.csv",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Produce a total research forecast for structurally valid future fixtures."""
    required = {"match_id", "kickoff_utc", "competition", "home_team", "away_team"}
    missing = sorted(required - set(fixtures.columns))
    if missing:
        raise RuntimeError(f"fallback forecast missing fixture columns: {missing}")

    f = fixtures.copy()
    f["match_id"] = f["match_id"].astype(str).str.strip()
    f["competition"] = f["competition"].astype(str).str.strip().str.upper()
    f["kickoff_utc"] = pd.to_datetime(f["kickoff_utc"], utc=True, errors="coerce")
    f["home_team"] = f["home_team"].astype(str).str.strip()
    f["away_team"] = f["away_team"].astype(str).str.strip()
    f = f[
        f["match_id"].ne("")
        & f["kickoff_utc"].notna()
        & f["home_team"].ne("")
        & f["away_team"].ne("")
        & (f["kickoff_utc"] > prediction_time)
    ].copy()

    history, history_meta = _load_pit_history(history_path, prediction_time)
    team_rates = _team_rates(history, prediction_time)
    comp_means = _competition_means(history, prediction_time)
    elo_cache: dict[tuple[str, str, str, bool], tuple[np.ndarray, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for row in f.itertuples(index=False):
        neutral = (
            bool(getattr(row, "neutral_venue"))
            if hasattr(row, "neutral_venue") and not pd.isna(getattr(row, "neutral_venue"))
            else str(row.competition).upper() in NEUTRAL_COMPETITIONS
        )
        elo_key = (
            str(row.home_team),
            str(row.away_team),
            str(row.competition),
            bool(neutral),
        )
        if elo_key not in elo_cache:
            elo_cache[elo_key] = _elo_probabilities(
                history,
                str(row.home_team),
                str(row.away_team),
                str(row.competition),
                bool(neutral),
            )
        elo_probabilities, elo_support = elo_cache[elo_key]

        market_probabilities = None
        market_verified = False
        if all(hasattr(row, name) for name in (
            "matchday_market_p_home",
            "matchday_market_p_draw",
            "matchday_market_p_away",
            "matchday_pit_verified",
            "matchday_available_at_utc",
        )):
            available_at = pd.to_datetime(
                getattr(row, "matchday_available_at_utc"), utc=True, errors="coerce"
            )
            verified = str(getattr(row, "matchday_pit_verified")).strip().lower() in {"true", "1", "yes"}
            values = np.asarray([
                getattr(row, "matchday_market_p_home"),
                getattr(row, "matchday_market_p_draw"),
                getattr(row, "matchday_market_p_away"),
            ], dtype=float)
            if (
                verified
                and pd.notna(available_at)
                and available_at <= prediction_time
                and np.isfinite(values).all()
                and np.all((values >= 0.0) & (values <= 1.0))
                and float(values.sum()) > 0.0
            ):
                market_probabilities = tuple(float(v) for v in values)
                market_verified = True

        prediction = predict_fallback_fixture(
            home_team=row.home_team,
            away_team=row.away_team,
            competition=row.competition,
            history=history,
            prediction_time=prediction_time,
            neutral_venue=neutral,
            team_rates=team_rates,
            comp_means=comp_means,
            market_probabilities=market_probabilities,
            elo_probabilities=tuple(float(v) for v in elo_probabilities),
        )
        scores = prediction["scores"]
        rows.append({
            "match_id": str(row.match_id),
            "kickoff_utc": pd.Timestamp(row.kickoff_utc).isoformat(),
            "competition": str(row.competition),
            "home_team": str(row.home_team),
            "away_team": str(row.away_team),
            "home_win_probability": prediction["p_home"],
            "draw_probability": prediction["p_draw"],
            "away_win_probability": prediction["p_away"],
            "result_prediction": prediction["result_prediction"],
            "result_prediction_probability": prediction["result_prediction_probability"],
            "score_1": scores[0]["score"],
            "score_1_probability": scores[0]["probability"],
            "score_2": scores[1]["score"],
            "score_2_probability": scores[1]["probability"],
            "score_3": scores[2]["score"],
            "score_3_probability": scores[2]["probability"],
            "over_2_5_probability": prediction["over_2_5_probability"],
            "under_2_5_probability": prediction["under_2_5_probability"],
            "btts_yes_probability": prediction["btts_yes_probability"],
            "btts_no_probability": prediction["btts_no_probability"],
            "mom_status": "BLOCKED_UPSTREAM_PLAYER_MODEL",
            "mom_method": "fallback_total_coverage_no_player_probability_source",
            "mom_1_player": "",
            "mom_1_probability": np.nan,
            "mom_2_player": "",
            "mom_2_probability": np.nan,
            "mom_3_player": "",
            "mom_3_probability": np.nan,
            "mom_4_player": "",
            "mom_4_probability": np.nan,
            "forecast_mode": "FALLBACK_BASELINE",
            "fallback_data_level": prediction["evidence"]["data_level"],
            "fallback_history_rows": int(history_meta.get("rows_pit_eligible", 0)),
            "fallback_competition_history_rows": int(prediction["evidence"]["competition_history_rows"]),
            "fallback_home_team_history_rows": int(prediction["evidence"]["home_team_history_rows"]),
            "fallback_away_team_history_rows": int(prediction["evidence"]["away_team_history_rows"]),
            "fallback_home_lambda": prediction["home_lambda"],
            "fallback_away_lambda": prediction["away_lambda"],
            "fallback_market_prior_used": bool(prediction["evidence"].get("market_prior_used", False)),
            "fallback_market_prior_weight": float(prediction["evidence"].get("market_weight", 0.0)),
            "fallback_market_pit_verified": bool(market_verified),
            "fallback_elo_prior_used": bool(prediction["evidence"].get("elo_prior_used", False)),
            "fallback_elo_weight": float(prediction["evidence"].get("elo_weight", 0.0)),
            "fallback_home_elo": float(elo_support.get("home_rating", 1500.0)),
            "fallback_away_elo": float(elo_support.get("away_rating", 1500.0)),
            "fallback_history_support_min": int(
                min(prediction["evidence"].get("home_team_history_rows", 0), prediction["evidence"].get("away_team_history_rows", 0))
            ),
            "fallback_entropy": float(
                -sum(
                    max(float(p), 1e-12) * math.log(max(float(p), 1e-12))
                    for p in (prediction["p_home"], prediction["p_draw"], prediction["p_away"])
                ) / math.log(3.0)
            ),
        })

    output = pd.DataFrame(rows)
    meta = {
        **history_meta,
        "forecast_mode": "FALLBACK_BASELINE",
        "fixtures_considered": int(len(f)),
        "prediction_rows": int(len(output)),
        "research_only": True,
        "production_registry_changed": False,
        "production_probabilities_changed": False,
        "fallback_always_total": True,
    }
    return output, meta
