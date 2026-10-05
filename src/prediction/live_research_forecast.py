from __future__ import annotations

"""Research-only current-match forecast. Never Production."""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

ELO_URL = "https://www.eloratings.net/World.tsv"
LIVE_URL = "https://www.sofascore.com/api/v1/sport/football/events/live"
SCHEDULE_URL = "https://www.sofascore.com/api/v1/sport/football/scheduled-events/{date}"
HOME_ADVANTAGE_ELO = 80.0
BASE_TOTAL_GOALS = 2.45
PROB_SHRINK = 0.85
TIMEOUT = 20.0


def now_utc():
    return pd.Timestamp(datetime.now(timezone.utc))


def get_text(url: str):
    at = now_utc().isoformat()
    r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "Soccer-Prediction-Research/1.0"})
    r.raise_for_status()
    return r.text, at


def get_json(url: str):
    at = now_utc().isoformat()
    r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "Soccer-Prediction-Research/1.0"})
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError("unexpected JSON payload")
    return data, at


def load_elo():
    text, at = get_text(ELO_URL)
    frame = pd.read_csv(pd.io.common.StringIO(text), sep="\t", engine="python")
    cols = {str(c).strip().lower(): c for c in frame.columns}
    team_col = next((cols[c] for c in ("team", "country", "name") if c in cols), None)
    rating_col = next((cols[c] for c in ("rating", "elo", "value") if c in cols), None)
    if team_col is None or rating_col is None:
        if len(frame.columns) < 3:
            raise RuntimeError("unsupported Elo TSV schema")
        team_col, rating_col = frame.columns[0], frame.columns[2]
    ratings = {}
    for _, row in frame.iterrows():
        team = str(row.get(team_col, "")).strip().lower()
        try:
            rating = float(row.get(rating_col))
        except (TypeError, ValueError):
            continue
        if team and np.isfinite(rating):
            ratings[team.replace("türkiye", "turkey")] = rating
    for alias, canonical in (("turkiye", "turkey"), ("türkiye", "turkey")):
        if alias in ratings and canonical not in ratings:
            ratings[canonical] = ratings[alias]
    required = ("france", "belgium", "italy", "turkey")
    missing = [x for x in required if x not in ratings]
    if missing:
        raise RuntimeError(f"Elo source missing teams: {missing}")
    return ratings, at


def find_target(event):
    home = str((event.get("homeTeam") or {}).get("name") or "").strip()
    away = str((event.get("awayTeam") or {}).get("name") or "").strip()
    pair = {home.lower().replace("türkiye", "turkey"), away.lower().replace("türkiye", "turkey")}
    if pair == {"france", "belgium"}:
        return "France-Belgium"
    if pair == {"italy", "turkey"}:
        return "Italy-Turkey"
    return None


def collect_events(ts):
    found = {}
    source_times = {}
    live, at = get_json(LIVE_URL)
    source_times["sofascore_live"] = at
    for event in live.get("events", []) or []:
        if isinstance(event, dict):
            key = find_target(event)
            if key:
                found[key] = event
    for day in (ts.date(), (ts + pd.Timedelta(days=1)).date()):
        payload, at = get_json(SCHEDULE_URL.format(date=day.isoformat()))
        source_times[f"sofascore_scheduled_{day.isoformat()}"] = at
        for event in payload.get("events", []) or []:
            if not isinstance(event, dict):
                continue
            key = find_target(event)
            if not key:
                continue
            kickoff = pd.to_datetime(event.get("startTimestamp"), unit="s", utc=True, errors="coerce")
            if pd.notna(kickoff) and kickoff >= ts - pd.Timedelta(minutes=5):
                found[key] = event
    return list(found.values()), source_times


def poisson(lam, max_goals=8):
    out = np.array([math.exp(-lam) * lam**g / math.factorial(g) for g in range(max_goals + 1)], dtype=float)
    return out / max(float(out.sum()), 1e-12)


def matrix(home_lambda, away_lambda, max_goals=8):
    out = np.outer(poisson(home_lambda, max_goals), poisson(away_lambda, max_goals))
    return out / max(float(out.sum()), 1e-12)


def result_probs(mat):
    raw = np.array([np.tril(mat, -1).sum(), np.trace(mat), np.triu(mat, 1).sum()], dtype=float)
    raw = raw / max(float(raw.sum()), 1e-12)
    out = PROB_SHRINK * raw + (1.0 - PROB_SHRINK) / 3.0
    return out / max(float(out.sum()), 1e-12)


def lambdas(home_elo, away_elo):
    delta = home_elo - away_elo + HOME_ADVANTAGE_ELO
    share = 1.0 / (1.0 + 10.0 ** (-delta / 400.0))
    return BASE_TOTAL_GOALS * share, BASE_TOTAL_GOALS * (1.0 - share)


def status_type(event):
    return str((event.get("status") or {}).get("type") or "").lower().strip()


def current_score(event):
    hs = event.get("homeScore") or {}
    aas = event.get("awayScore") or {}
    try:
        h = int(hs.get("normaltime", hs.get("current", 0)) or 0)
        a = int(aas.get("normaltime", aas.get("current", 0)) or 0)
    except (TypeError, ValueError):
        h, a = 0, 0
    return max(0, h), max(0, a)


def elapsed(event, ts):
    kickoff = pd.to_datetime(event.get("startTimestamp"), unit="s", utc=True, errors="coerce")
    if pd.isna(kickoff):
        return 0.0
    return float(np.clip((ts - kickoff).total_seconds() / 60.0, 0.0, 90.0))


def live_matrix(home_elo, away_elo, hg, ag, mins):
    bh, ba = lambdas(home_elo, away_elo)
    f = float(np.clip(mins / 90.0, 0.0, 1.0))
    expected = max(BASE_TOTAL_GOALS * f, 0.05)
    observed = hg + ag
    tempo = float(np.clip(observed / expected, 0.65, 1.55))
    tempo = float(np.clip(0.70 + 0.30 * tempo, 0.80, 1.30))
    rem = max(0.0, 1.0 - f)
    rh, ra = max(1e-6, bh * rem * tempo), max(1e-6, ba * rem * tempo)
    remaining = matrix(rh, ra, 8)
    final = np.zeros((9, 9), dtype=float)
    for h in range(9):
        for a in range(9):
            fh, fa = hg + h, ag + a
            if fh < 9 and fa < 9:
                final[fh, fa] += remaining[h, a]
    return final / max(float(final.sum()), 1e-12)


def top_scores(mat):
    pairs = [(f"{h}-{a}", float(mat[h, a])) for h in range(mat.shape[0]) for a in range(mat.shape[1]) if mat[h, a] > 0]
    pairs.sort(key=lambda x: (-x[1], x[0]))
    return pairs[:3]


def predict(event, ratings, ts, source_times):
    home = str((event.get("homeTeam") or {}).get("name") or "").strip()
    away = str((event.get("awayTeam") or {}).get("name") or "").strip()
    hk, ak = home.lower().replace("türkiye", "turkey"), away.lower().replace("türkiye", "turkey")
    kickoff = pd.to_datetime(event.get("startTimestamp"), unit="s", utc=True, errors="coerce")
    if pd.isna(kickoff):
        raise RuntimeError("invalid kickoff")
    if hk not in ratings or ak not in ratings:
        raise RuntimeError(f"missing Elo for {home} vs {away}")
    hgoals, agoals = current_score(event)
    mins = elapsed(event, ts)
    st = status_type(event)
    live = st in {"inprogress", "live", "halftime"} or (kickoff <= ts < kickoff + pd.Timedelta(minutes=100) and hgoals + agoals > 0)
    if live:
        mat = live_matrix(ratings[hk], ratings[ak], hgoals, agoals, mins)
        state = "LIVE_RESEARCH_FORECAST"
        pit = "CURRENT_OBSERVED_BEFORE_CUTOFF"
    else:
        if kickoff <= ts:
            raise RuntimeError("past event not eligible for pre-match prediction")
        lh, la = lambdas(ratings[hk], ratings[ak])
        mat = matrix(lh, la)
        state = "PREMATCH_RESEARCH_FORECAST"
        pit = "CURRENT_OBSERVED_PRE_KICKOFF"
    probs = result_probs(mat)
    labels = ("Home", "Draw", "Away")
    top = top_scores(mat)
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
    predictability = float(np.clip(1.0 - entropy / math.log(3), 0.0, 1.0))
    return {
        "match_id": f"sofascore:{event.get('id')}",
        "source_event_id": str(event.get("id") or ""),
        "event_source": "sofascore",
        "kickoff_utc": kickoff.isoformat(),
        "prediction_time_utc": ts.isoformat(),
        "home_team": home,
        "away_team": away,
        "competition": str(((event.get("tournament") or {}).get("uniqueTournament") or {}).get("name") or ""),
        "match_status": st or "unknown",
        "elapsed_minutes": round(mins, 2),
        "current_home_goals": hgoals,
        "current_away_goals": agoals,
        "home_elo": ratings[hk],
        "away_elo": ratings[ak],
        "result_prediction": labels[int(np.argmax(probs))],
        "p_home": round(float(probs[0]), 6),
        "p_draw": round(float(probs[1]), 6),
        "p_away": round(float(probs[2]), 6),
        "score_1": top[0][0], "score_1_probability": round(top[0][1], 6),
        "score_2": top[1][0], "score_2_probability": round(top[1][1], 6),
        "score_3": top[2][0], "score_3_probability": round(top[2][1], 6),
        "uncertainty": round(1.0 - float(np.max(probs)), 6),
        "predictability": round(predictability, 6),
        "prediction_state": state,
        "model_status": "RESEARCH_ONLY_HEURISTIC_ELO_POISSON",
        "production_status": "NOT_PRODUCTION",
        "pit_status": pit,
        "source_available_at_utc": None,
        "source_available_lower_bound_utc": source_times.get("sofascore_live"),
        "source_retrieved_at_utc": max(source_times.values()),
        "elo_source_url": ELO_URL,
        "event_source_url": f"https://www.sofascore.com/event/{event.get('id')}",
        "calibration_method": "0.85_model_plus_0.15_uniform_shrink",
    }


REQUIRED = {
    "match_id", "kickoff_utc", "home_team", "away_team", "result_prediction",
    "p_home", "p_draw", "p_away", "score_1", "score_1_probability",
    "score_2", "score_2_probability", "score_3", "score_3_probability",
    "prediction_state", "pit_status", "production_status",
}


def verify(path):
    df = pd.read_csv(path)
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        raise RuntimeError(f"missing fields: {missing}")
    errors = []
    for i, row in df.iterrows():
        p = np.asarray([row.p_home, row.p_draw, row.p_away], dtype=float)
        if not np.isfinite(p).all() or not np.isclose(p.sum(), 1.0, atol=1e-5):
            errors.append(f"{i}: probability sum")
        if row.production_status != "NOT_PRODUCTION":
            errors.append(f"{i}: production flag")
        if row.pit_status not in {"CURRENT_OBSERVED_BEFORE_CUTOFF", "CURRENT_OBSERVED_PRE_KICKOFF"}:
            errors.append(f"{i}: PIT status")
        for n in (1, 2, 3):
            sp = float(row[f"score_{n}_probability"])
            if "-" not in str(row[f"score_{n}"]) or not np.isfinite(sp) or not 0 <= sp <= 1:
                errors.append(f"{i}: score top{n}")
    if errors:
        raise RuntimeError("; ".join(errors[:20]))
    return {"status": "VERIFIED", "rows": int(len(df))}


def run(output, status):
    ts = now_utc()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(status).parent.mkdir(parents=True, exist_ok=True)
    try:
        ratings, elo_at = load_elo()
        events, source_times = collect_events(ts)
        source_times["eloratings_world_tsv"] = elo_at
        rows, failures = [], []
        for event in events:
            try:
                rows.append(predict(event, ratings, ts, source_times))
            except Exception as exc:
                failures.append(f"{type(exc).__name__}: {exc}")
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.sort_values(["kickoff_utc", "home_team"], kind="mergesort")
        frame.to_csv(output, index=False)
        payload = {
            "status": "PREDICTED_LIVE_RESEARCH" if rows else "NO_TARGET_EVENT_OR_BLOCKED",
            "prediction_time_utc": ts.isoformat(),
            "rows": len(rows),
            "failures": failures,
            "model_status": "RESEARCH_ONLY_HEURISTIC_ELO_POISSON",
            "production_status": "NOT_PRODUCTION",
            "source_times": source_times,
            "pit_note": "Retrieval is kept distinct from source availability; historical publication availability is not inferred.",
        }
        Path(status).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return payload
    except Exception as exc:
        pd.DataFrame().to_csv(output, index=False)
        payload = {"status": "FAILED_CLOSED", "prediction_time_utc": ts.isoformat(), "rows": 0, "error": f"{type(exc).__name__}: {exc}", "production_status": "NOT_PRODUCTION"}
        Path(status).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/live_research_predictions.csv")
    parser.add_argument("--status", default="artifacts/live_research_prediction_status.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = run(args.output, args.status)
    if args.verify:
        result["output_contract_verification"] = verify(args.output)
        result["output_contract_verified"] = True
        Path(args.status).write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") != "FAILED_CLOSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
