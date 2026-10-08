from __future__ import annotations

"""Current-match research forecast lane.

This module is deliberately outside the Production adoption path. It provides a
fresh, current-state research forecast for explicitly requested matches while
preserving provenance, timestamps, uncertainty and fail-closed behavior.
"""

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

DEFAULT_CONFIG_PATH = Path("config/live_research_forecast.json")

ESPN_FALLBACK_LEAGUES = (
    "fifa.friendly",
    "fifa.worldq.uefa",
    "uefa.nations",
    "fifa.world",
)

REQUIRED = {
    "match_id",
    "prediction_revision_id",
    "kickoff_utc",
    "home_team",
    "away_team",
    "result_prediction",
    "p_home",
    "p_draw",
    "p_away",
    "score_1",
    "score_1_probability",
    "score_2",
    "score_2_probability",
    "score_3",
    "score_3_probability",
    "prediction_state",
    "pit_status",
    "production_status",
    "data_completeness",
    "uncertainty",
    "predictability",
    "source_snapshot_hash",
    "source_snapshot_hashes",
    "config_sha256",
    "git_commit_sha",
    "experiment_fingerprint",
}


def now_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def load_config() -> tuple[dict[str, Any], str]:
    path = Path(os.environ.get("LIVE_FORECAST_CONFIG", str(DEFAULT_CONFIG_PATH)))
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"live forecast config missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
        raise RuntimeError("unsupported live forecast config schema")
    return payload, hashlib.sha256(path.read_bytes()).hexdigest()


def _get(url: str, timeout: float) -> tuple[bytes, str, str]:
    at = now_utc().isoformat()
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "Soccer-Prediction-Research/1.0"},
    )
    response.raise_for_status()
    body = response.content
    return body, at, hashlib.sha256(body).hexdigest()


def _get_json(url: str, timeout: float) -> tuple[dict[str, Any], str, str]:
    body, at, digest = _get(url, timeout)
    data = json.loads(body.decode("utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"unexpected JSON payload from {url}")
    return data, at, digest


def parse_elo_tsv(text: str, elo_codes: dict[str, str]) -> dict[str, float]:
    """Parse current eloratings World.tsv conservatively.

    World.tsv is a compact tab-separated table without a stable header contract;
    the current layout exposes team code in column 3 and Elo in column 4.
    """
    ratings_by_code: dict[str, float] = {}
    for raw in text.splitlines():
        cells = raw.strip().split("\t")
        if len(cells) < 4:
            continue
        code = cells[2].strip().upper()
        try:
            rating = float(cells[3])
        except (TypeError, ValueError):
            continue
        if code and np.isfinite(rating):
            ratings_by_code[code] = rating
    ratings = {
        team_key: ratings_by_code[code]
        for code, team_key in elo_codes.items()
        if code in ratings_by_code
    }
    return ratings


def load_elo(config: dict[str, Any]) -> tuple[dict[str, float], str, str]:
    timeout = float(config["runtime"]["request_timeout_seconds"])
    body, observed_at, digest = _get(
        str(config["sources"]["elo_url"]),
        timeout,
    )
    ratings = parse_elo_tsv(body.decode("utf-8", errors="replace"), config["team_aliases"]["elo_codes"])
    missing = [team for team in config["target_teams"] if team not in ratings]
    if missing:
        raise RuntimeError(f"Elo source missing teams: {missing}")
    return ratings, observed_at, digest


def _normalized_team(value: str) -> str:
    return " ".join(value.strip().lower().replace("türkiye", "turkey").split())


def find_target(event: dict[str, Any], config: dict[str, Any]) -> str | None:
    home = str((event.get("homeTeam") or {}).get("name") or "").strip()
    away = str((event.get("awayTeam") or {}).get("name") or "").strip()
    if not home or not away:
        return None
    observed = frozenset((_normalized_team(home), _normalized_team(away)))
    for label, pair in config["target_pairs"].items():
        wanted = frozenset(_normalized_team(x) for x in pair)
        if observed == wanted:
            return str(label)
    return None


def _espn_event_adapter(event: dict[str, Any], league: str) -> dict[str, Any] | None:
    competitions = event.get("competitions") or []
    competition = competitions[0] if competitions and isinstance(competitions[0], dict) else {}
    competitors = competition.get("competitors") or []
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if not isinstance(home, dict) or not isinstance(away, dict):
        return None
    home_team = (home.get("team") or {}) if isinstance(home.get("team"), dict) else {}
    away_team = (away.get("team") or {}) if isinstance(away.get("team"), dict) else {}
    home_name = str(home_team.get("displayName") or home_team.get("name") or "").strip()
    away_name = str(away_team.get("displayName") or away_team.get("name") or "").strip()
    kickoff = pd.to_datetime(event.get("date"), utc=True, errors="coerce")
    if not home_name or not away_name or pd.isna(kickoff) or event.get("id") is None:
        return None
    def score_of(item: dict[str, Any]) -> int:
        value = item.get("score")
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0
    status_type = (event.get("status") or {}).get("type") or {}
    state = str(status_type.get("state") or "").strip().lower()
    name = str(status_type.get("name") or "").strip().lower()
    if state in {"in", "inprogress"} or "halftime" in name:
        status = "inprogress" if "halftime" not in name else "halftime"
    elif state in {"post", "postponed", "canceled"}:
        status = state
    else:
        status = "scheduled"
    return {
        "id": f"espn:{league}:{event.get('id')}",
        "startTimestamp": int(kickoff.timestamp()),
        "homeTeam": {"name": home_name},
        "awayTeam": {"name": away_name},
        "status": {"type": status},
        "homeScore": {"current": score_of(home)},
        "awayScore": {"current": score_of(away)},
        "tournament": {"uniqueTournament": {"name": league}},
    }


def collect_events(
    prediction_time: pd.Timestamp,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str]]:
    timeout = float(config["runtime"]["request_timeout_seconds"])
    found: dict[str, tuple[dict[str, Any], str]] = {}
    source_times: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    sofascore_failed = False

    try:
        live, observed, digest = _get_json(str(config["sources"]["sofascore_live_url"]), timeout)
        source_times["sofascore_live"] = observed
        source_hashes["sofascore_live"] = digest
        for event in live.get("events", []) or []:
            if not isinstance(event, dict):
                continue
            label = find_target(event, config)
            if label:
                found[label] = (event, "sofascore")
    except Exception as exc:
        sofascore_failed = True
        source_times["sofascore_live_error"] = now_utc().isoformat()
        source_hashes["sofascore_live_error"] = f"{type(exc).__name__}:{exc}"

    schedule_template = str(config["sources"]["sofascore_scheduled_url"])
    schedule_days = max(int(config["runtime"]["schedule_days"]), 1)
    for offset in range(schedule_days):
        day = (prediction_time + pd.Timedelta(days=offset)).date()
        key = f"sofascore_scheduled_{day.isoformat()}"
        try:
            payload, observed, digest = _get_json(schedule_template.format(date=day.isoformat()), timeout)
            source_times[key] = observed
            source_hashes[key] = digest
            for event in payload.get("events", []) or []:
                if not isinstance(event, dict):
                    continue
                label = find_target(event, config)
                if not label:
                    continue
                kickoff = pd.to_datetime(event.get("startTimestamp"), unit="s", utc=True, errors="coerce")
                if pd.isna(kickoff):
                    continue
                if kickoff >= prediction_time - pd.Timedelta(minutes=float(config["runtime"]["schedule_past_tolerance_minutes"])):
                    found.setdefault(label, (event, "sofascore"))
        except Exception:
            source_times[f"{key}_error"] = now_utc().isoformat()

    if sofascore_failed or not found:
        for league in config.get("sources", {}).get("espn_fallback_leagues", list(ESPN_FALLBACK_LEAGUES)):
            for offset in range(min(schedule_days, 2)):
                day = (prediction_time + pd.Timedelta(days=offset)).strftime("%Y%m%d")
                key = f"espn_{league}_{day}"
                url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard?dates={day}"
                try:
                    payload, observed, digest = _get_json(url, timeout)
                    source_times[key] = observed
                    source_hashes[key] = digest
                    for raw_event in payload.get("events", []) or []:
                        if not isinstance(raw_event, dict):
                            continue
                        event = _espn_event_adapter(raw_event, league)
                        if not event:
                            continue
                        label = find_target(event, config)
                        if not label:
                            continue
                        kickoff = pd.to_datetime(event.get("startTimestamp"), unit="s", utc=True, errors="coerce")
                        if pd.isna(kickoff) or kickoff < prediction_time - pd.Timedelta(minutes=float(config["runtime"]["schedule_past_tolerance_minutes"])):
                            continue
                        found.setdefault(label, (event, "espn"))
                except Exception as exc:
                    source_times[f"{key}_error"] = now_utc().isoformat()
                    source_hashes[f"{key}_error"] = f"{type(exc).__name__}:{exc}"

    return [event for event, _ in found.values()], source_times, source_hashes


def poisson(lam: float, max_goals: int) -> np.ndarray:
    values = np.asarray(
        [math.exp(-lam) * lam**g / math.factorial(g) for g in range(max_goals + 1)],
        dtype=float,
    )
    total = float(values.sum())
    return values / max(total, 1e-12)


def score_matrix(home_lambda: float, away_lambda: float, max_goals: int) -> np.ndarray:
    matrix = np.outer(poisson(home_lambda, max_goals), poisson(away_lambda, max_goals))
    return matrix / max(float(matrix.sum()), 1e-12)


def result_probs(matrix: np.ndarray, shrink: float) -> np.ndarray:
    raw = np.asarray(
        [np.tril(matrix, -1).sum(), np.trace(matrix), np.triu(matrix, 1).sum()],
        dtype=float,
    )
    raw = raw / max(float(raw.sum()), 1e-12)
    shrink = float(np.clip(shrink, 0.0, 1.0))
    out = shrink * raw + (1.0 - shrink) / 3.0
    return out / max(float(out.sum()), 1e-12)


def pre_match_lambdas(home_elo: float, away_elo: float, config: dict[str, Any]) -> tuple[float, float]:
    model = config["model"]
    delta = home_elo - away_elo + float(model["home_advantage_elo"])
    share = 1.0 / (1.0 + 10.0 ** (-delta / 400.0))
    total = float(model["base_total_goals"])
    return total * share, total * (1.0 - share)


def _status_type(event: dict[str, Any]) -> str:
    return str((event.get("status") or {}).get("type") or "").strip().lower()


def _current_score(event: dict[str, Any]) -> tuple[int, int]:
    home_score = event.get("homeScore") or {}
    away_score = event.get("awayScore") or {}
    try:
        home = int(home_score.get("normaltime", home_score.get("current", 0)) or 0)
        away = int(away_score.get("normaltime", away_score.get("current", 0)) or 0)
    except (TypeError, ValueError):
        home, away = 0, 0
    return max(0, home), max(0, away)


def _elapsed(event: dict[str, Any], prediction_time: pd.Timestamp) -> float:
    kickoff = pd.to_datetime(event.get("startTimestamp"), unit="s", utc=True, errors="coerce")
    if pd.isna(kickoff):
        return 0.0
    return float(
        np.clip((prediction_time - kickoff).total_seconds() / 60.0, 0.0, 90.0)
    )


def live_final_matrix(
    home_elo: float,
    away_elo: float,
    current_home: int,
    current_away: int,
    elapsed_minutes: float,
    config: dict[str, Any],
) -> np.ndarray:
    model = config["model"]
    base_home, base_away = pre_match_lambdas(home_elo, away_elo, config)
    fraction = float(np.clip(elapsed_minutes / 90.0, 0.0, 1.0))
    expected = max(float(model["base_total_goals"]) * fraction, 0.05)
    observed = current_home + current_away
    raw_tempo = observed / expected
    tempo = float(
        np.clip(
            raw_tempo,
            float(model["tempo_ratio_min"]),
            float(model["tempo_ratio_max"]),
        )
    )
    tempo_multiplier = float(
        np.clip(
            float(model["tempo_multiplier_base"]) + float(model["tempo_multiplier_slope"]) * tempo,
            float(model["tempo_multiplier_min"]),
            float(model["tempo_multiplier_max"]),
        )
    )
    remaining_fraction = max(0.0, 1.0 - fraction)
    rem_home = max(1e-6, base_home * remaining_fraction * tempo_multiplier)
    rem_away = max(1e-6, base_away * remaining_fraction * tempo_multiplier)
    remaining = score_matrix(rem_home, rem_away, int(model["max_goals"]))
    size = int(model["max_goals"]) + 1
    final = np.zeros((size, size), dtype=float)
    for rh in range(remaining.shape[0]):
        for ra in range(remaining.shape[1]):
            fh, fa = current_home + rh, current_away + ra
            if fh < size and fa < size:
                final[fh, fa] += remaining[rh, ra]
    return final / max(float(final.sum()), 1e-12)


def _top_scores(matrix: np.ndarray, n: int) -> list[tuple[str, float]]:
    values = [
        (f"{home}-{away}", float(matrix[home, away]))
        for home in range(matrix.shape[0])
        for away in range(matrix.shape[1])
        if matrix[home, away] > 0
    ]
    values.sort(key=lambda x: (-x[1], x[0]))
    return values[:n]


def predict(
    event: dict[str, Any],
    ratings: dict[str, float],
    prediction_time: pd.Timestamp,
    source_times: dict[str, str],
    source_hashes: dict[str, str],
    config: dict[str, Any],
    config_hash: str,
    elo_hash: str,
    event_source: str = "sofascore",
) -> dict[str, Any]:
    home = str((event.get("homeTeam") or {}).get("name") or "").strip()
    away = str((event.get("awayTeam") or {}).get("name") or "").strip()
    home_key = _normalized_team(home)
    away_key = _normalized_team(away)
    kickoff = pd.to_datetime(event.get("startTimestamp"), unit="s", utc=True, errors="coerce")
    if pd.isna(kickoff):
        raise RuntimeError(f"invalid kickoff: {home} vs {away}")
    if home_key not in ratings or away_key not in ratings:
        raise RuntimeError(f"missing Elo rating: {home} vs {away}")

    status = _status_type(event)
    current_home, current_away = _current_score(event)
    elapsed_minutes = _elapsed(event, prediction_time)
    live_window = kickoff <= prediction_time < kickoff + pd.Timedelta(minutes=100)
    live = status in {"inprogress", "live", "halftime"} or (live_window and (current_home + current_away > 0))

    if live:
        matrix = live_final_matrix(
            ratings[home_key],
            ratings[away_key],
            current_home,
            current_away,
            elapsed_minutes,
            config,
        )
        prediction_state = "LIVE_RESEARCH_FORECAST"
        pit_status = "CURRENT_OBSERVED_BEFORE_CUTOFF"
    else:
        if kickoff <= prediction_time:
            raise RuntimeError(f"past event is not eligible: {home} vs {away}")
        home_lambda, away_lambda = pre_match_lambdas(ratings[home_key], ratings[away_key], config)
        matrix = score_matrix(home_lambda, away_lambda, int(config["model"]["max_goals"]))
        prediction_state = "PREMATCH_RESEARCH_FORECAST"
        pit_status = "CURRENT_OBSERVED_PRE_KICKOFF"

    probs = result_probs(matrix, float(config["model"]["probability_shrink"]))
    labels = ("Home", "Draw", "Away")
    top = _top_scores(matrix, int(config["model"]["top_scorelines"]))

    required_state_fields = (
        event.get("id"),
        event.get("startTimestamp"),
        home,
        away,
        event.get("status"),
        event.get("homeScore"),
        event.get("awayScore"),
    )
    data_completeness = float(
        sum(value is not None for value in required_state_fields) / len(required_state_fields)
    )
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
    base_predictability = float(np.clip(1.0 - entropy / math.log(3.0), 0.0, 1.0))
    predictability = float(np.clip(base_predictability * data_completeness, 0.0, 1.0))
    uncertainty = float(
        np.clip(
            1.0 - float(np.max(probs)) + (1.0 - data_completeness) * float(config["model"]["data_quality_uncertainty_penalty"]),
            0.0,
            1.0,
        )
    )

    event_hash = hashlib.sha256(
        json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    git_sha = os.environ.get("GITHUB_SHA", "UNKNOWN")
    fingerprint = hashlib.sha256(
        "|".join([config_hash, git_sha, elo_hash, event_hash, prediction_state]).encode("utf-8")
    ).hexdigest()
    revision_id = fingerprint[:24]

    source_snapshot_hashes = dict(source_hashes)
    return {
        "match_id": f"{event_source}:{event.get('id')}",
        "prediction_revision_id": revision_id,
        "event_source": event_source,
        "source_event_id": str(event.get("id") or ""),
        "kickoff_utc": pd.Timestamp(kickoff).isoformat(),
        "prediction_time_utc": prediction_time.isoformat(),
        "home_team": home,
        "away_team": away,
        "competition": str(((event.get("tournament") or {}).get("uniqueTournament") or {}).get("name") or ""),
        "match_status": status or "unknown",
        "elapsed_minutes": round(elapsed_minutes, 2),
        "current_home_goals": current_home,
        "current_away_goals": current_away,
        "home_elo": float(ratings[home_key]),
        "away_elo": float(ratings[away_key]),
        "result_prediction": labels[int(np.argmax(probs))],
        "p_home": round(float(probs[0]), 6),
        "p_draw": round(float(probs[1]), 6),
        "p_away": round(float(probs[2]), 6),
        "score_1": top[0][0],
        "score_1_probability": round(top[0][1], 6),
        "score_2": top[1][0],
        "score_2_probability": round(top[1][1], 6),
        "score_3": top[2][0],
        "score_3_probability": round(top[2][1], 6),
        "uncertainty": round(uncertainty, 6),
        "predictability": round(predictability, 6),
        "data_completeness": round(data_completeness, 6),
        "prediction_state": prediction_state,
        "decision_state": "PREDICT",
        "model_status": "RESEARCH_ONLY_HEURISTIC_ELO_POISSON",
        "production_status": "NOT_PRODUCTION",
        "pit_status": pit_status,
        "source_available_at_utc": None,
        "source_available_lower_bound_utc": source_times.get("sofascore_live"),
        "source_retrieved_at_utc": max(source_times.values()),
        "source_snapshot_hash": event_hash,
        "source_snapshot_hashes": json.dumps(source_snapshot_hashes, sort_keys=True),
        "config_sha256": config_hash,
        "git_commit_sha": git_sha,
        "experiment_fingerprint": fingerprint,
        "elo_source_url": str(config["sources"]["elo_url"]),
        "event_source_url": (f"https://www.sofascore.com/event/{event.get('id')}" if event_source == "sofascore" else f"https://www.espn.com/soccer/match/_/gameId/{event.get('id')}"),
        "calibration_method": "uniform_shrink_only_research_heuristic",
    }


def empty_output(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=sorted(REQUIRED)).to_csv(path, index=False)


def run(output: str, status_path: str) -> dict[str, Any]:
    prediction_time = now_utc()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(status_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        config, config_hash = load_config()
        ratings, elo_at, elo_hash = load_elo(config)
        events, source_times, source_hashes = collect_events(prediction_time, config)
        source_times["eloratings_world_tsv"] = elo_at
        source_hashes["eloratings_world_tsv"] = elo_hash

        rows: list[dict[str, Any]] = []
        failures: list[str] = []
        for event in events:
            try:
                source = "espn" if str(event.get("id", "")).startswith("espn:") else "sofascore"
                rows.append(
                    predict(
                        event,
                        ratings,
                        prediction_time,
                        source_times,
                        source_hashes,
                        config,
                        config_hash,
                        elo_hash,
                        event_source=source,
                    )
                )
            except Exception as exc:
                failures.append(f"{type(exc).__name__}: {exc}")

        if rows:
            frame = pd.DataFrame(rows).sort_values(["kickoff_utc", "home_team"], kind="mergesort")
        else:
            frame = pd.DataFrame(columns=sorted(REQUIRED))
        frame.to_csv(output, index=False)

        status = {
            "status": "PREDICTED_LIVE_RESEARCH" if rows else "NO_TARGET_EVENT_OR_BLOCKED",
            "prediction_time_utc": prediction_time.isoformat(),
            "rows": len(rows),
            "failures": failures,
            "model_status": "RESEARCH_ONLY_HEURISTIC_ELO_POISSON",
            "production_status": "NOT_PRODUCTION",
            "decision_policy": "PREDICT_ONLY_WHEN_CURRENT_EVENT_AND_REQUIRED_STATE_ARE_OBSERVED; OTHERWISE_DEFER",
            "config_sha256": config_hash,
            "git_commit_sha": os.environ.get("GITHUB_SHA", "UNKNOWN"),
            "source_times": source_times,
            "source_hashes": source_hashes,
            "pit_note": "Retrieval is not treated as historical publication availability; source_available_at_utc remains unknown.",
        }
        Path(status_path).write_text(
            json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return status
    except Exception as exc:
        empty_output(output)
        status = {
            "status": "FAILED_CLOSED",
            "prediction_time_utc": prediction_time.isoformat(),
            "rows": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "production_status": "NOT_PRODUCTION",
        }
        Path(status_path).write_text(
            json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return status


def verify(path: str) -> dict[str, Any]:
    df = pd.read_csv(path)
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        raise RuntimeError(f"missing fields: {missing}")
    errors: list[str] = []
    for idx, row in df.iterrows():
        probs = np.asarray([row["p_home"], row["p_draw"], row["p_away"]], dtype=float)
        if not np.isfinite(probs).all() or not np.isclose(probs.sum(), 1.0, atol=1e-5):
            errors.append(f"{idx}: invalid 1X2 probability sum")
        if str(row["production_status"]) != "NOT_PRODUCTION":
            errors.append(f"{idx}: production flag must remain NOT_PRODUCTION")
        if str(row["pit_status"]) not in {"CURRENT_OBSERVED_BEFORE_CUTOFF", "CURRENT_OBSERVED_PRE_KICKOFF"}:
            errors.append(f"{idx}: invalid PIT status")
        if not 0.0 <= float(row["uncertainty"]) <= 1.0:
            errors.append(f"{idx}: uncertainty out of range")
        if not 0.0 <= float(row["predictability"]) <= 1.0:
            errors.append(f"{idx}: predictability out of range")
        for rank in (1, 2, 3):
            score = str(row[f"score_{rank}"])
            probability = float(row[f"score_{rank}_probability"])
            if "-" not in score or not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                errors.append(f"{idx}: invalid score top{rank}")
    if errors:
        raise RuntimeError("; ".join(errors[:20]))
    return {"status": "VERIFIED", "rows": int(len(df)), "empty_target_set": bool(df.empty)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/live_research_predictions.csv")
    parser.add_argument("--status", default="artifacts/live_research_prediction_status.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    status = run(args.output, args.status)
    if args.verify:
        verification = verify(args.output)
        status["output_contract_verification"] = verification
        status["output_contract_verified"] = verification.get("status") == "VERIFIED"
        Path(args.status).write_text(
            json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(status, ensure_ascii=False))
    return 0 if status.get("status") != "FAILED_CLOSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
