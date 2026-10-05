"""Prospective, PIT-safe in-play capture and maturity pipeline.

This module never rewrites the past into a stronger historical claim. Live state is
captured when directly observed from a free/public current endpoint, and label
maturity is derived only from later observations of the same prospective capture.
No market odds are stored or consumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SOURCE = "ESPN_PUBLIC_SOCCER"
USER_AGENT = "Soccer-Prediction-Research/prospective-inplay/1.0"
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard?dates={date}"
LEAGUES = {
    "eng.1": "EPL",
    "esp.1": "LaLiga",
    "ita.1": "SerieA",
    "ger.1": "Bundesliga",
    "fra.1": "Ligue1",
    "ned.1": "Eredivisie",
    "por.1": "PrimeiraLiga",
    "bel.1": "Belgium",
    "tur.1": "Turkey",
    "uefa.champions": "UCL",
    "uefa.europa": "UEL",
}
LIVE_STATES = {"in", "in_progress"}
COMPLETED_STATES = {"post", "final"}
HAZARD_WINDOW_MINUTES = 5.0


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: Any) -> datetime | None:
    if value in (None, "", "null"):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc)


def _request_json(url: str, *, timeout: int = 15, retries: int = 3) -> tuple[dict[str, Any], str]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
            },
            method="GET",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read()
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON response is not an object")
            return payload, hashlib.sha256(raw).hexdigest()
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(8, attempt * 2))
    raise RuntimeError(f"public endpoint fetch failed: {type(last_error).__name__}: {last_error}")


def _event_competition(event: dict[str, Any]) -> dict[str, Any]:
    competitions = event.get("competitions")
    if not isinstance(competitions, list) or not competitions:
        return {}
    first = competitions[0]
    return first if isinstance(first, dict) else {}


def _competitors(event: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    comp = _event_competition(event)
    values = comp.get("competitors")
    if not isinstance(values, list):
        return None, None
    home = next((x for x in values if isinstance(x, dict) and x.get("homeAway") == "home"), None)
    away = next((x for x in values if isinstance(x, dict) and x.get("homeAway") == "away"), None)
    return home, away


def _team_info(competitor: dict[str, Any] | None) -> tuple[str, str]:
    if not competitor:
        return "", ""
    team = competitor.get("team") if isinstance(competitor.get("team"), dict) else {}
    return str(team.get("id", competitor.get("id", ""))), str(
        team.get("displayName", competitor.get("displayName", ""))
    )


def _score(competitor: dict[str, Any] | None) -> float | None:
    if not competitor:
        return None
    try:
        return float(competitor.get("score"))
    except (TypeError, ValueError):
        return None


def _status(event: dict[str, Any]) -> tuple[str, float | None, int | None]:
    comp = _event_competition(event)
    status = comp.get("status")
    if not isinstance(status, dict):
        status = event.get("status") if isinstance(event.get("status"), dict) else {}
    status_type = status.get("type") if isinstance(status.get("type"), dict) else {}
    state = str(status_type.get("state", "")).strip().lower()
    try:
        clock = float(status.get("clock")) if status.get("clock") is not None else None
    except (TypeError, ValueError):
        clock = None
    try:
        period = int(status.get("period")) if status.get("period") is not None else None
    except (TypeError, ValueError):
        period = None
    return state, clock, period


def _play_text(play: dict[str, Any]) -> str:
    parts = [
        play.get("text"),
        (play.get("type") or {}).get("text") if isinstance(play.get("type"), dict) else None,
        (play.get("type") or {}).get("description") if isinstance(play.get("type"), dict) else None,
    ]
    return " ".join(str(v) for v in parts if v).strip().lower()


def _play_team_id(play: dict[str, Any]) -> str:
    team = play.get("team")
    if isinstance(team, dict):
        return str(team.get("id", ""))
    return str(play.get("teamId", ""))


def _play_clock_minutes(play: dict[str, Any]) -> float | None:
    clock = play.get("clock")
    if isinstance(clock, dict):
        raw = clock.get("value")
        try:
            seconds = float(raw)
            period = int(play.get("period", {}).get("number", 1))
            return max(0.0, (45.0 if period >= 2 else 0.0) + seconds / 60.0)
        except (TypeError, ValueError):
            pass
    return None


def _summary_plays(summary: dict[str, Any]) -> list[dict[str, Any]]:
    plays = summary.get("plays")
    if isinstance(plays, list):
        return [p for p in plays if isinstance(p, dict)]
    scoring = summary.get("scoringPlays")
    if isinstance(scoring, list):
        return [p for p in scoring if isinstance(p, dict)]
    return []


def _red_card_state(
    summary: dict[str, Any],
    *,
    cutoff_clock_minutes: float | None,
    home_id: str,
    away_id: str,
) -> tuple[int | None, int | None, bool]:
    plays = _summary_plays(summary)
    if not plays:
        return None, None, False
    home_red = 0
    away_red = 0
    parseable = True
    for play in plays:
        text = _play_text(play)
        is_red = "red card" in text or "sent off" in text
        if not is_red:
            continue
        minute = _play_clock_minutes(play)
        if minute is None or cutoff_clock_minutes is None:
            parseable = False
            continue
        if minute > cutoff_clock_minutes + 1e-9:
            continue
        team_id = _play_team_id(play)
        if team_id == home_id:
            home_red += 1
        elif team_id == away_id:
            away_red += 1
        else:
            parseable = False
    if not parseable:
        return None, None, False
    return home_red, away_red, True


def parse_live_snapshot(
    *,
    event: dict[str, Any],
    summary: dict[str, Any],
    league: str,
    observed_at: datetime,
    response_sha256: str,
) -> dict[str, Any] | None:
    state, clock_seconds, period = _status(event)
    if state not in LIVE_STATES and state not in COMPLETED_STATES:
        return None
    home, away = _competitors(event)
    home_id, home_name = _team_info(home)
    away_id, away_name = _team_info(away)
    home_score = _score(home)
    away_score = _score(away)
    kickoff = _parse_time(
        _event_competition(event).get("startDate")
        or event.get("date")
    )
    if not home_id or not away_id or kickoff is None or home_score is None or away_score is None:
        return None

    clock_minutes = clock_seconds / 60.0 if clock_seconds is not None else None
    home_red, away_red, red_known = _red_card_state(
        summary,
        cutoff_clock_minutes=clock_minutes,
        home_id=home_id,
        away_id=away_id,
    )
    row = {
        "schema_version": 1,
        "source_name": SOURCE,
        "source_url": SCOREBOARD_URL.format(league=league, date=observed_at.strftime("%Y%m%d")),
        "event_id": str(event.get("id", "")),
        "competition": LEAGUES.get(league, league),
        "espn_league": league,
        "kickoff_utc": kickoff.isoformat(),
        "observed_at_utc": observed_at.isoformat(),
        "prediction_cutoff_utc": observed_at.isoformat(),
        "source_available_at_utc": observed_at.isoformat(),
        "published_at_utc": None,
        "retrieved_at_utc": observed_at.isoformat(),
        "revision_time_utc": None,
        "event_time_utc": observed_at.isoformat(),
        "pit_verified": True,
        "home_team_id": home_id,
        "away_team_id": away_id,
        "home_team": home_name,
        "away_team": away_name,
        "home_score": home_score,
        "away_score": away_score,
        "home_red_cards": home_red,
        "away_red_cards": away_red,
        "red_cards_known": bool(red_known),
        "status_state": state,
        "clock_seconds": clock_seconds,
        "period": period,
        "hazard_window_minutes": HAZARD_WINDOW_MINUTES,
        "source_response_sha256": response_sha256,
        "capture_slot_utc": observed_at.replace(second=0, microsecond=0, minute=(observed_at.minute // 2) * 2).isoformat(),
        "label_available_at_utc": None,
        "next_event_type": None,
        "next_event_time_utc": None,
        "final_home_goals": None,
        "final_away_goals": None,
    }
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    row["snapshot_fingerprint"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return row


def _daily_path(root: Path, day: datetime) -> Path:
    path = root / day.strftime("%Y-%m-%d.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _existing_fingerprints(path: Path, *, limit: int = 1000) -> set[str]:
    if not path.is_file():
        return set()
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    values = set()
    for line in lines:
        try:
            item = json.loads(line)
            if item.get("snapshot_fingerprint"):
                values.add(str(item["snapshot_fingerprint"]))
        except json.JSONDecodeError:
            continue
    return values


def capture_once(
    *,
    output_root: str | Path = "data/research/prospective_inplay/raw",
    observed_at: datetime | None = None,
    leagues: Iterable[str] = LEAGUES.keys(),
) -> dict[str, Any]:
    now = observed_at.astimezone(timezone.utc) if observed_at else utcnow()
    path = _daily_path(Path(output_root), now)
    existing = _existing_fingerprints(path)
    existing_completed: set[str] = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines()[-5000:]:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(item.get("status_state", "")).lower() in COMPLETED_STATES:
                existing_completed.add(str(item.get("event_id", "")))
    captured: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for league in leagues:
        url = SCOREBOARD_URL.format(league=league, date=now.strftime("%Y%m%d"))
        try:
            scoreboard, response_hash = _request_json(url)
        except Exception as exc:
            errors.append({"league": league, "error": f"{type(exc).__name__}: {exc}"})
            continue
        events = scoreboard.get("events")
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            state, _, _ = _status(event)
            if state not in LIVE_STATES and state not in COMPLETED_STATES:
                continue
            event_id = str(event.get("id", ""))
            if state in COMPLETED_STATES and event_id in existing_completed:
                continue
            if state in COMPLETED_STATES:
                kickoff_probe = _parse_time(
                    _event_competition(event).get("startDate") or event.get("date")
                )
                if kickoff_probe is None or (now - kickoff_probe).total_seconds() > 135 * 60:
                    continue
            summary_url = (
                f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/summary"
                f"?event={event.get('id', '')}"
            )
            try:
                summary, _ = _request_json(summary_url)
                row = parse_live_snapshot(
                    event=event,
                    summary=summary,
                    league=league,
                    observed_at=now,
                    response_sha256=response_hash,
                )
            except Exception as exc:
                errors.append({
                    "league": league,
                    "event_id": str(event.get("id", "")),
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            if row and row["snapshot_fingerprint"] not in existing:
                captured.append(row)
                existing.add(row["snapshot_fingerprint"])

    if captured:
        with path.open("a", encoding="utf-8") as handle:
            for row in captured:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\\n")

    return {
        "schema_version": 1,
        "status": "CAPTURED",
        "observed_at_utc": now.isoformat(),
        "path": str(path),
        "captured_rows": len(captured),
        "captured_matches": len({r["event_id"] for r in captured}),
        "errors": errors,
        "source": SOURCE,
        "odds_consumed": False,
        "pit_method": "prospective_direct_observation",
    }


def capture_loop(
    *,
    output_root: str | Path = "data/research/prospective_inplay/raw",
    loops: int = 1,
    interval_seconds: int = 0,
) -> dict[str, Any]:
    if loops < 1 or loops > 12:
        raise ValueError("loops must be in [1, 12]")
    if interval_seconds < 0 or interval_seconds > 900:
        raise ValueError("interval_seconds must be in [0, 900]")
    reports = []
    for i in range(loops):
        reports.append(capture_once(output_root=output_root))
        if i + 1 < loops and interval_seconds:
            time.sleep(interval_seconds)
    return {
        "schema_version": 1,
        "status": "CAPTURE_LOOP_COMPLETE",
        "loops": loops,
        "interval_seconds": interval_seconds,
        "reports": reports,
        "captured_rows": sum(r["captured_rows"] for r in reports),
        "errors": sum(len(r["errors"]) for r in reports),
        "odds_consumed": False,
    }


def _load_raw(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for path in sorted(root.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get("event_id"):
                rows.append(item)
    return rows


def _same_state(a: dict[str, Any], b: dict[str, Any]) -> tuple[int, str | None]:
    fields = [
        ("home_score", "HOME_GOAL"),
        ("away_score", "AWAY_GOAL"),
        ("home_red_cards", "HOME_RED"),
        ("away_red_cards", "AWAY_RED"),
    ]
    deltas = []
    for field, event_name in fields:
        av = a.get(field)
        bv = b.get(field)
        if av is None or bv is None:
            return 0, None
        try:
            delta = int(round(float(bv) - float(av)))
        except (TypeError, ValueError):
            return 0, None
        if delta < 0 or delta > 1:
            return 0, None
        if delta == 1:
            deltas.append(event_name)
    if len(deltas) == 0:
        return 0, "NO_EVENT"
    if len(deltas) == 1:
        return 1, deltas[0]
    return 0, None


def build_mature_rows(
    raw_rows: Iterable[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    current = now.astimezone(timezone.utc) if now else utcnow()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[str(row["event_id"])].append(dict(row))

    output: list[dict[str, Any]] = []
    for event_id, rows in grouped.items():
        rows.sort(key=lambda x: x.get("observed_at_utc", ""))
        completed = [
            r for r in rows
            if str(r.get("status_state", "")).lower() in COMPLETED_STATES
            and r.get("home_score") is not None
            and r.get("away_score") is not None
        ]
        if not completed:
            continue
        final_row = completed[-1]
        final_home = float(final_row["home_score"])
        final_away = float(final_row["away_score"])
        final_observed = _parse_time(final_row.get("observed_at_utc"))
        if final_observed is None or final_observed > current:
            continue

        for idx, snap in enumerate(rows):
            if str(snap.get("status_state", "")).lower() not in LIVE_STATES:
                continue
            if not snap.get("red_cards_known"):
                continue
            if idx + 1 >= len(rows):
                continue
            nxt = rows[idx + 1]
            t0 = _parse_time(snap.get("observed_at_utc"))
            t1 = _parse_time(nxt.get("observed_at_utc"))
            if t0 is None or t1 is None:
                continue
            gap = (t1 - t0).total_seconds() / 60.0
            if gap <= 0 or gap > HAZARD_WINDOW_MINUTES:
                continue

            magnitude, label = _same_state(snap, nxt)
            if label is None:
                continue

            row = dict(snap)
            row["schema_version"] = 2
            row["label_available_at_utc"] = t1.isoformat()
            row["next_event_type"] = label
            row["next_event_time_utc"] = (
                t1.isoformat() if label != "NO_EVENT" else None
            )
            row["final_home_goals"] = final_home
            row["final_away_goals"] = final_away
            row["final_label_available_at_utc"] = final_observed.isoformat()
            row["maturity_method"] = "prospective_state_transition"
            row["maturity_note"] = (
                "next_event_time_utc is the first later observation proving a state transition; "
                "it is not a claim of exact event publication time"
            )
            row["matured_at_utc"] = current.isoformat()
            row["event_magnitude"] = magnitude
            output.append(row)

    output.sort(key=lambda x: (str(x["kickoff_utc"]), str(x["event_id"]), str(x["prediction_cutoff_utc"])))
    return output


def write_monthly_matured(
    raw_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    raw_rows = _load_raw(Path(raw_root))
    mature = build_mature_rows(raw_rows)
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_month: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in mature:
        kickoff = _parse_time(row.get("kickoff_utc"))
        if kickoff:
            by_month[kickoff.strftime("%Y-%m")].append(row)

    written = []
    for month, rows in by_month.items():
        path = out_dir / f"{month}.csv"
        import csv
        fieldnames = sorted({k for row in rows for k in row})
        existing_keys = set()
        if path.is_file():
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for item in reader:
                    existing_keys.add(
                        (item.get("event_id", ""), item.get("prediction_cutoff_utc", ""))
                    )
        new_rows = [
            row for row in rows
            if (str(row.get("event_id", "")), str(row.get("prediction_cutoff_utc", ""))) not in existing_keys
        ]
        if not new_rows:
            continue
        mode = "a" if path.is_file() else "w"
        with path.open(mode, encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            if mode == "w":
                writer.writeheader()
            for row in new_rows:
                writer.writerow(row)
        written.append({"month": month, "path": str(path), "new_rows": len(new_rows)})

    return {
        "schema_version": 1,
        "status": "MATURED",
        "raw_rows": len(raw_rows),
        "matured_rows": len(mature),
        "written": written,
        "odds_consumed": False,
        "pit_method": "prospective_direct_observation_then_later_maturity",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture")
    capture.add_argument("--output-root", default="data/research/prospective_inplay/raw")
    capture.add_argument("--loops", type=int, default=1)
    capture.add_argument("--interval-seconds", type=int, default=0)
    mature = sub.add_parser("mature")
    mature.add_argument("--raw-root", default="data/research/prospective_inplay/raw")
    mature.add_argument("--output-root", default="data/research/match_state_snapshots")
    args = parser.parse_args()
    if args.command == "capture":
        print(json.dumps(capture_loop(
            output_root=args.output_root,
            loops=args.loops,
            interval_seconds=args.interval_seconds,
        ), ensure_ascii=False))
    else:
        print(json.dumps(write_monthly_matured(
            args.raw_root,
            args.output_root,
        ), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
