from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.data.competition_sources import TARGET_COMPETITIONS
from src.data.football_data import load_available_history
from src.data.matchday_intelligence_fetch import ESPN_LEAGUES
from src.evaluation.score import score_distribution


POSITION_PRIOR = {
    "F": 1.00, "FW": 1.00, "ST": 1.00, "CF": 1.00,
    "AM": 0.92, "W": 0.90, "M": 0.82, "MF": 0.82,
    "D": 0.52, "DF": 0.52, "FB": 0.52, "CB": 0.48,
    "GK": 0.22,
}


def _now_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def _fit_goal_rates(history: pd.DataFrame, prediction_time: pd.Timestamp) -> dict[str, Any]:
    d = history.copy()
    for col in ("kickoff_utc", "home_goals", "away_goals"):
        if col not in d.columns:
            raise RuntimeError(f"Historical data missing {col}")
    d["kickoff_utc"] = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    d["home_goals"] = pd.to_numeric(d["home_goals"], errors="coerce")
    d["away_goals"] = pd.to_numeric(d["away_goals"], errors="coerce")
    d = d.dropna(subset=["kickoff_utc", "home_goals", "away_goals", "home_team", "away_team"])
    d = d[d["kickoff_utc"] < prediction_time].sort_values("kickoff_utc", kind="mergesort")
    if d.empty:
        raise RuntimeError("No historical results before prediction_time")

    shrink = 20.0
    home_mean = float(d["home_goals"].mean())
    away_mean = float(d["away_goals"].mean())
    overall_mean = float((d["home_goals"].sum() + d["away_goals"].sum()) / (2.0 * len(d)))

    home = d.assign(_team=d["home_team"].astype(str)).groupby("_team").agg(
        n=("home_goals", "size"),
        scored=("home_goals", "sum"),
        conceded=("away_goals", "sum"),
    )
    away = d.assign(_team=d["away_team"].astype(str)).groupby("_team").agg(
        n=("away_goals", "size"),
        scored=("away_goals", "sum"),
        conceded=("home_goals", "sum"),
    )
    teams: dict[str, dict[str, float]] = {}
    for team in sorted(set(home.index.astype(str)) | set(away.index.astype(str))):
        h = home.loc[team] if team in home.index else None
        a = away.loc[team] if team in away.index else None
        hn = float(h["n"]) if h is not None else 0.0
        an = float(a["n"]) if a is not None else 0.0
        hs = float(h["scored"]) if h is not None else 0.0
        hc = float(h["conceded"]) if h is not None else 0.0
        ass = float(a["scored"]) if a is not None else 0.0
        ac = float(a["conceded"]) if a is not None else 0.0
        teams[team] = {
            "home_attack": (hs + shrink * home_mean) / (hn + shrink),
            "home_defence": (hc + shrink * away_mean) / (hn + shrink),
            "away_attack": (ass + shrink * away_mean) / (an + shrink),
            "away_defence": (ac + shrink * home_mean) / (an + shrink),
        }

    comp_rates: dict[str, tuple[float, float]] = {}
    if "competition" in d.columns:
        for comp, g in d.groupby(d["competition"].astype(str), sort=True):
            n = len(g)
            comp_rates[str(comp)] = (
                float((g["home_goals"].sum() + shrink * home_mean) / (n + shrink)),
                float((g["away_goals"].sum() + shrink * away_mean) / (n + shrink)),
            )
    return {
        "home_mean": home_mean,
        "away_mean": away_mean,
        "overall_mean": overall_mean,
        "teams": teams,
        "comp_rates": comp_rates,
    }


def _lambdas(model: dict[str, Any], home_team: str, away_team: str, competition: str) -> tuple[float, float]:
    base_h, base_a = model["comp_rates"].get(
        str(competition), (model["home_mean"], model["away_mean"])
    )
    h = model["teams"].get(str(home_team))
    a = model["teams"].get(str(away_team))
    if h is None:
        h = {"home_attack": base_h, "home_defence": base_a}
    if a is None:
        a = {"away_attack": base_a, "away_defence": base_h}
    lh = base_h * float(h["home_attack"]) / max(base_h, 1e-6) * float(a["away_defence"]) / max(base_h, 1e-6)
    la = base_a * float(a["away_attack"]) / max(base_a, 1e-6) * float(h["home_defence"]) / max(base_a, 1e-6)
    return float(np.clip(lh, 0.05, 5.0)), float(np.clip(la, 0.05, 5.0))


def _one_x_two(dist: list[tuple[int, int, float]]) -> tuple[float, float, float]:
    values = np.asarray([
        sum(p for h, a, p in dist if h > a),
        sum(p for h, a, p in dist if h == a),
        sum(p for h, a, p in dist if h < a),
    ], dtype=float)
    values /= values.sum()
    return tuple(float(x) for x in values)


def _market(row: pd.Series) -> tuple[float, float, float] | None:
    cols = ("matchday_market_p_home", "matchday_market_p_draw", "matchday_market_p_away")
    try:
        values = np.asarray([float(row.get(c)) for c in cols], dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(values).all() or values.min() < 0 or values.sum() <= 0:
        return None
    values /= values.sum()
    return tuple(float(x) for x in values)


def _position_score(position: str) -> float:
    p = str(position or "").upper()
    for key, value in POSITION_PRIOR.items():
        if p == key or key in p:
            return value
    return 0.65


def _mom_candidates(row: pd.Series, session: requests.Session) -> tuple[list[tuple[str, float]], str]:
    league = ESPN_LEAGUES.get(str(row.get("competition")))
    if not league:
        return [], "BLOCKED_NO_ROSTER_ENDPOINT"
    players: dict[str, float] = {}
    for side in ("home_team_id", "away_team_id"):
        team_id = str(row.get(side) or "").strip()
        if not team_id:
            continue
        url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/teams/{team_id}/roster"
        try:
            response = session.get(
                url, timeout=15, headers={"User-Agent": "SoccerPredictionResearch/1.0"}
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            continue
        athletes = payload.get("athletes") or payload.get("items") or []
        for athlete in athletes:
            if not isinstance(athlete, dict):
                continue
            name = str(athlete.get("displayName") or athlete.get("fullName") or "").strip()
            if not name:
                continue
            position = athlete.get("position") or {}
            abbr = str(position.get("abbreviation") or position.get("name") or "")
            players[name] = max(players.get(name, 0.0), _position_score(abbr))
    if len(players) < 4:
        return [], "BLOCKED_FEWER_THAN_4_PLAYERS"
    ranked = sorted(players.items(), key=lambda item: (-item[1], item[0]))[:4]
    raw = np.asarray([score for _, score in ranked], dtype=float)
    raw /= raw.sum()
    return [(name, float(prob)) for (name, _), prob in zip(ranked, raw)], "PREDICTED_HEURISTIC"


def run(fixtures_path: str, output_path: str, status_path: str, prediction_time: str | None = None) -> dict[str, Any]:
    now = pd.Timestamp(prediction_time) if prediction_time else _now_utc()
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")

    fixtures = pd.read_csv(fixtures_path)
    required = {"match_id", "kickoff_utc", "home_team", "away_team", "competition"}
    missing = sorted(required - set(fixtures.columns))
    if missing:
        raise RuntimeError(f"fixture snapshot missing columns: {missing}")
    fixtures["kickoff_utc"] = pd.to_datetime(fixtures["kickoff_utc"], utc=True, errors="coerce")
    fixtures["competition"] = fixtures["competition"].astype(str).str.strip().str.upper()
    fixtures = fixtures[
        fixtures["competition"].isin(TARGET_COMPETITIONS)
        & (fixtures["kickoff_utc"] > now)
    ].copy()
    fixtures = fixtures.sort_values(["kickoff_utc", "match_id"], kind="mergesort").reset_index(drop=True)

    history, _ = load_available_history()
    model = _fit_goal_rates(history, now)
    session = requests.Session()
    rows: list[dict[str, Any]] = []

    for _, row in fixtures.iterrows():
        lambdas = _lambdas(model, str(row["home_team"]), str(row["away_team"]), str(row["competition"]))
        distribution = score_distribution(*lambdas, max_goals=10)
        model_probs = _one_x_two(distribution)
        market = _market(row)
        probs = np.asarray(model_probs, dtype=float)
        if market is not None:
            probs = 0.70 * probs + 0.30 * np.asarray(market, dtype=float)
        probs /= probs.sum()
        score_top = distribution[:3]
        labels = [str(row["home_team"]), "引き分け", str(row["away_team"])]
        winner_index = int(np.argmax(probs))
        moms, mom_status = _mom_candidates(row, session)

        out: dict[str, Any] = {
            "match_id": str(row["match_id"]),
            "kickoff_utc": row["kickoff_utc"].isoformat(),
            "competition": str(row["competition"]),
            "home_team": str(row["home_team"]),
            "away_team": str(row["away_team"]),
            "home_win_probability": float(probs[0]),
            "draw_probability": float(probs[1]),
            "away_win_probability": float(probs[2]),
            "result_prediction": labels[winner_index],
            "result_prediction_probability": float(probs[winner_index]),
            "score_1": f"{score_top[0][0]}-{score_top[0][1]}",
            "score_1_probability": float(score_top[0][2]),
            "score_2": f"{score_top[1][0]}-{score_top[1][1]}",
            "score_2_probability": float(score_top[1][2]),
            "score_3": f"{score_top[2][0]}-{score_top[2][1]}",
            "score_3_probability": float(score_top[2][2]),
            "mom_status": mom_status,
            "mom_method": "roster_position_prior_research_only",
        }
        for rank in range(1, 5):
            out[f"mom_{rank}_player"] = moms[rank - 1][0] if len(moms) >= rank else ""
            out[f"mom_{rank}_probability"] = float(moms[rank - 1][1]) if len(moms) >= rank else np.nan
        rows.append(out)

    result = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    status = {
        "status": "PREDICTED_RESEARCH_ONLY" if not result.empty else "NO_TARGET_FIXTURES",
        "prediction_time_utc": now.isoformat(),
        "rows": int(len(result)),
        "model": "chronological_goal_rate_baseline_70pct_plus_market_30pct_when_available",
        "production_model_used": False,
        "production_adoption_bypassed": False,
        "mom_policy": "research_only_roster_position_prior",
        "mom_predicted_rows": int(result["mom_status"].astype(str).str.startswith("PREDICTED").sum()) if not result.empty else 0,
    }
    Path(status_path).write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return status


def verify(path: str) -> dict[str, Any]:
    df = pd.read_csv(path)
    errors: list[str] = []
    for idx, row in df.iterrows():
        probs = np.asarray([
            float(row["home_win_probability"]),
            float(row["draw_probability"]),
            float(row["away_win_probability"]),
        ])
        if not np.isfinite(probs).all() or not np.isclose(probs.sum(), 1.0, atol=1e-6):
            errors.append(f"{idx}: invalid 1X2 probabilities")
        for rank in (1, 2, 3):
            score = str(row.get(f"score_{rank}", ""))
            p = float(row.get(f"score_{rank}_probability", np.nan))
            if "-" not in score or not np.isfinite(p) or not 0.0 <= p <= 1.0:
                errors.append(f"{idx}: invalid score top{rank}")
        if str(row.get("mom_status", "")).startswith("PREDICTED"):
            mom_names = [str(row.get(f"mom_{r}_player", "")).strip() for r in (1,2,3,4)]
            mom_probs = np.asarray([float(row[f"mom_{r}_probability"]) for r in (1,2,3,4)])
            if any(not name for name in mom_names):
                errors.append(f"{idx}: incomplete MOM top4")
            if len(set(mom_names)) != 4:
                errors.append(f"{idx}: duplicate MOM candidates")
            if not np.isfinite(mom_probs).all() or np.any(mom_probs < 0) or not np.isclose(mom_probs.sum(), 1.0, atol=1e-6):
                errors.append(f"{idx}: invalid MOM probabilities")
    if errors:
        raise RuntimeError("; ".join(errors[:20]))
    return {"status": "VERIFIED", "rows": int(len(df))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--output", default="artifacts/daily_research_predictions.csv")
    parser.add_argument("--status", default="artifacts/daily_research_prediction_status.json")
    parser.add_argument("--prediction-time", default=None)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    status = run(args.fixtures, args.output, args.status, args.prediction_time)
    if args.verify:
        status.update(verify(args.output))
        Path(args.status).write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
