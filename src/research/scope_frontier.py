"""Continuously discover soccer competitions outside the current mapped scope.

Discovery is intentionally research-only: an unseen tournament becomes a
frontier candidate, never an automatic production target.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from src.data.competition_sources import TARGET_COMPETITIONS
from src.data.matchday_intelligence_fetch import _sofascore_competition

SOFASCORE_URL = "https://www.sofascore.com/api/v1/sport/football/scheduled-events/{date}"


def _fetch_json(day: date, timeout: float = 20.0) -> dict[str, Any]:
    request = Request(
        SOFASCORE_URL.format(date=day.isoformat()),
        headers={
            "User-Agent": "Soccer-Prediction-Research/1.0",
            "Accept": "application/json",
            "Referer": "https://www.sofascore.com/",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("SofaScore response root must be an object")
    return payload


def _tournament_identity(event: dict[str, Any]) -> tuple[str, str, str]:
    tournament = event.get("tournament") or {}
    unique = tournament.get("uniqueTournament") or event.get("uniqueTournament") or {}
    name = str(unique.get("name") or tournament.get("name") or "").strip()
    slug = str(unique.get("slug") or tournament.get("slug") or "").strip().lower()
    tid = str(unique.get("id") or tournament.get("id") or "").strip()
    return name, slug, tid


def _candidate(identity: tuple[str, str, str], *, first_seen: str, events: int) -> dict[str, Any]:
    name, slug, tid = identity
    key = tid or slug or name.lower()
    lower = name.lower()
    if any(token in lower for token in ("women", "female", "ladies")):
        case_type = "women"
    elif any(token in lower for token in ("u17", "u18", "u19", "u20", "u21", "u23", "youth")):
        case_type = "youth"
    elif any(token in lower for token in ("cup", "copa", "pokal")):
        case_type = "cup"
    elif "friendly" in lower:
        case_type = "friendly"
    else:
        case_type = "league_or_international"
    return {
        "id": f"sofascore:{key}",
        "target": name,
        "type": case_type,
        "rationale": "Observed by public SofaScore scheduled-events discovery but not mapped to a known competition code.",
        "eligibility": "RESEARCH_ONLY_UNVERIFIED",
        "volume": events,
        "production_value": "UNKNOWN",
        "learning_value": "HIGH",
        "coverage_value": "HIGH",
        "novelty": "HIGH",
        "data_availability": "OBSERVED",
        "quality": "UNVERIFIED",
        "pit_feasibility": "UNVERIFIED",
        "acquisition_reliability": "UNVERIFIED",
        "oos_risk": "HIGH",
        "false_discovery_risk": "MEDIUM",
        "operational_risk": "LOW",
        "cost": "FREE",
        "dependency_risk": "LOW",
        "reversibility": "HIGH",
        "stage": "DISCOVERED",
        "evidence": {
            "source": "SofaScore public scheduled-events",
            "first_seen_utc_date": first_seen,
            "tournament_id": tid,
            "slug": slug,
        },
        "next_test": "metadata/source validation -> PIT validation -> historical acquisition -> chronological OOS",
        "review_trigger": "new evidence, new source, repeated coverage, or mapped alias",
    }


def discover(days: int, output: str, *, start: date | None = None) -> dict[str, Any]:
    start = start or date.today()
    known = set(TARGET_COMPETITIONS)
    observed_known: dict[str, int] = {}
    discovered: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    scanned_dates: list[str] = []

    for offset in range(max(1, int(days))):
        day = start + timedelta(days=offset)
        scanned_dates.append(day.isoformat())
        try:
            payload = _fetch_json(day)
        except Exception as exc:
            errors.append({"date": day.isoformat(), "error": f"{type(exc).__name__}: {exc}"})
            continue
        for event in payload.get("events", []) or []:
            if not isinstance(event, dict):
                continue
            mapped = _sofascore_competition(event)
            if mapped in known:
                observed_known[mapped] = observed_known.get(mapped, 0) + 1
                continue
            identity = _tournament_identity(event)
            if not any(identity):
                continue
            name, slug, tid = identity
            key = tid or slug or name.lower()
            if key not in discovered:
                discovered[key] = _candidate(identity, first_seen=day.isoformat(), events=0)
            discovered[key]["volume"] = int(discovered[key]["volume"]) + 1

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous: dict[str, Any] = {}
    if path.is_file() and path.stat().st_size > 0:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                previous = payload
        except Exception:
            previous = {}

    merged: dict[str, dict[str, Any]] = {}
    old_items = previous.get("discovered_candidates")
    if isinstance(old_items, list):
        for item in old_items:
            if isinstance(item, dict) and item.get("id"):
                merged[str(item["id"])] = item
    for item in discovered.values():
        old = merged.get(item["id"], {})
        old.update(item)
        merged[item["id"]] = old

    result = {
        "schema_version": 1,
        "status": "OK" if not errors else "PARTIAL",
        "scanned_from_utc_date": scanned_dates[0] if scanned_dates else None,
        "scanned_to_utc_date": scanned_dates[-1] if scanned_dates else None,
        "scanned_calendar_days": len(scanned_dates),
        "known_target_event_counts": dict(sorted(observed_known.items())),
        "discovered_candidates": sorted(
            merged.values(), key=lambda x: (str(x.get("target", "")), str(x.get("id", "")))
        ),
        "new_candidates_this_run": len(discovered),
        "errors": errors,
        "production_auto_promotion": False,
        "fail_closed_on_unknown_pit": True,
    }
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--output", default="docs/scope_frontier.json")
    parser.add_argument("--start-date", default=None)
    args = parser.parse_args()
    start = date.fromisoformat(args.start_date) if args.start_date else None
    result = discover(max(1, args.days), args.output, start=start)
    print(json.dumps({
        "status": result["status"],
        "scanned_calendar_days": result["scanned_calendar_days"],
        "discovered_candidates": len(result["discovered_candidates"]),
        "new_candidates_this_run": result["new_candidates_this_run"],
        "errors": len(result["errors"]),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
