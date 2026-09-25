"""FotMob MOM-label acquisition for research-only training labels.

FotMob is used only as an outcome/label source in this module. Match discovery and
player-of-the-match extraction are separated from pre-match feature construction.
Missing or ambiguous labels fail closed. FotMob player IDs are provider-specific and
must be reconciled to the candidate dataset by exact normalized player name before join.
"""
from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from typing import Any

import pandas as pd

from src.data.external_fetch import ExternalFetcher


FOTMOB_BASE = "https://www.fotmob.com/api/data"
DEFAULT_FOTMOB_LEAGUE_ID = 47  # Premier League
DEFAULT_FOTMOB_SEASON = "2024/2025"
DEFAULT_MAX_MATCH_DELTA_HOURS = 6.0
FOTMOB_HEADERS = {
    "User-Agent": "SoccerPredictionResearch/1.0 MOM-label-audit",
    "Referer": "https://www.fotmob.com/",
}


def _norm_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    kept: list[str] = []
    for ch in text:
        if unicodedata.combining(ch):
            previous = kept[-1] if kept else ""
            if "LATIN" in unicodedata.name(previous, ""):
                continue
        kept.append(ch)
    text = unicodedata.normalize("NFKC", "".join(kept).casefold())
    return "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] not in {"P", "Z"}
    )


def _parse_json(body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"FotMob JSON decode failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("FotMob response root must be an object")
    return payload


def _name_from_team(obj: Any) -> str:
    if not isinstance(obj, dict):
        return ""
    for key in ("name", "teamName", "shortName", "longName"):
        value = obj.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _timestamp_from_match(obj: dict[str, Any]) -> pd.Timestamp | None:
    candidates = [
        obj.get("utcTime"),
        obj.get("startTime"),
        obj.get("kickoff"),
        (obj.get("status") or {}).get("utcTime") if isinstance(obj.get("status"), dict) else None,
        (obj.get("status") or {}).get("utcTimeStr") if isinstance(obj.get("status"), dict) else None,
        (obj.get("time") or {}).get("utcTime") if isinstance(obj.get("time"), dict) else None,
    ]
    for value in candidates:
        if value in (None, ""):
            continue
        # Numeric epochs must be interpreted explicitly; pandas otherwise treats
        # bare integers as nanoseconds, which can silently map real fixtures to 1970.
        try:
            numeric = float(value)
            if math.isfinite(numeric):
                unit = "ms" if abs(numeric) > 10_000_000_000 else "s"
                ts = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
                if not pd.isna(ts):
                    return ts
        except (TypeError, ValueError, OverflowError):
            pass
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if not pd.isna(ts):
            return ts
    return None


def _match_record(obj: dict[str, Any]) -> dict[str, Any] | None:
    general = obj.get("general")
    if isinstance(general, dict):
        match_id = general.get("matchId") or general.get("id") or obj.get("id")
        home = _name_from_team(general.get("homeTeam"))
        away = _name_from_team(general.get("awayTeam"))
        if match_id not in (None, "") and home and away:
            header = obj.get("header") if isinstance(obj.get("header"), dict) else {}
            status = header.get("status") if isinstance(header.get("status"), dict) else {}
            start_obj = dict(obj)
            start_obj["status"] = status
            kickoff = _timestamp_from_match(start_obj) or _timestamp_from_match(general)
            if kickoff is not None:
                return {
                    "match_id": str(match_id),
                    "home_team": home,
                    "away_team": away,
                    "kickoff_utc": kickoff,
                }

    match_id = obj.get("matchId") or obj.get("id")
    home = _name_from_team(obj.get("home"))
    away = _name_from_team(obj.get("away"))
    if match_id not in (None, "") and home and away:
        kickoff = _timestamp_from_match(obj)
        if kickoff is not None:
            return {
                "match_id": str(match_id),
                "home_team": home,
                "away_team": away,
                "kickoff_utc": kickoff,
            }
    return None


def _extract_matches(payload: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            record = _match_record(obj)
            if record is not None:
                key = (record["match_id"], _norm_text(record["home_team"]), _norm_text(record["away_team"]))
                if key not in seen:
                    seen.add(key)
                    found.append(record)
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    walk(payload)
    return found


def _extract_potm(payload: dict[str, Any]) -> tuple[str, str, str]:
    content = payload.get("content")
    if not isinstance(content, dict):
        return "", "", ""
    facts = content.get("matchFacts")
    if not isinstance(facts, dict):
        return "", "", ""
    potm = facts.get("playerOfTheMatch")
    if not isinstance(potm, dict):
        return "", "", ""
    player_id = potm.get("id")
    name_obj = potm.get("name")
    if isinstance(name_obj, dict):
        name = name_obj.get("fullName") or " ".join(
            str(x) for x in (name_obj.get("firstName"), name_obj.get("lastName")) if x
        )
    else:
        name = str(name_obj or potm.get("shortName") or "")
    team_name = ""
    team_obj = potm.get("team")
    if isinstance(team_obj, dict):
        team_name = _name_from_team(team_obj)
    return str(player_id or ""), name, team_name


def fetch_fotmob_league_matches(
    *,
    league_id: int,
    season: str,
    fetcher: ExternalFetcher,
) -> tuple[list[dict[str, Any]], str]:
    url = f"{FOTMOB_BASE}/leagues"
    response = fetcher.get(
        "fotmob_league_matches",
        url,
        params={"id": int(league_id), "ccode3": "ENG", "season": str(season)},
        headers=FOTMOB_HEADERS,
    )
    payload = _parse_json(response.body)
    matches = _extract_matches(payload)
    if not matches:
        raise RuntimeError("FotMob league response contained no match records")
    return matches, response.metadata.retrieved_at


def fetch_fotmob_match_potm(
    match_id: str | int,
    *,
    fetcher: ExternalFetcher,
) -> dict[str, Any]:
    url = f"{FOTMOB_BASE}/matchDetails"
    response = fetcher.get(
        "fotmob_match_details",
        url,
        params={"matchId": str(match_id)},
        headers=FOTMOB_HEADERS,
    )
    payload = _parse_json(response.body)
    player_id, player_name, player_team = _extract_potm(payload)
    return {
        "event_id": str(match_id),
        "player_provider_id": player_id,
        "player_name": player_name,
        "player_team_name": player_team,
        "label_found": bool(player_id or player_name),
        "retrieved_at_utc": response.metadata.retrieved_at,
        "source_content_sha256": hashlib.sha256(response.body).hexdigest(),
        "source": "fotmob_match_details_player_of_the_match",
        "cache_hit": bool(getattr(response.metadata, "cache_hit", False)),
    }


def collect_fotmob_mom_labels(
    fixtures: pd.DataFrame,
    *,
    league_id: int = DEFAULT_FOTMOB_LEAGUE_ID,
    season: str = DEFAULT_FOTMOB_SEASON,
    cache_dir: str = "cache/external",
    max_match_delta_hours: float = DEFAULT_MAX_MATCH_DELTA_HOURS,
    retries: int = 3,
    request_delay_seconds: float = 0.25,
) -> pd.DataFrame:
    required = {"match_id", "kickoff_utc", "home_team", "away_team"}
    missing = sorted(required - set(fixtures.columns))
    if missing:
        raise ValueError(f"FotMob label fixtures missing columns: {missing}")
    delta_limit = float(max_match_delta_hours)
    delay = float(request_delay_seconds)
    if not math.isfinite(delta_limit) or delta_limit < 0:
        raise ValueError("max_match_delta_hours must be finite and non-negative")
    if not math.isfinite(delay) or delay < 0:
        raise ValueError("request_delay_seconds must be finite and non-negative")

    d = fixtures.copy()
    d["match_id"] = d["match_id"].astype("string").str.strip()
    d["kickoff_utc"] = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    if d["kickoff_utc"].isna().any() or d["match_id"].eq("").any():
        raise ValueError("FotMob label fixtures contain invalid match_id/kickoff_utc")

    fetcher = ExternalFetcher(cache_dir=cache_dir, retries=retries)
    matches, league_retrieved_at = fetch_fotmob_league_matches(
        league_id=int(league_id),
        season=str(season),
        fetcher=fetcher,
    )

    indexed: list[dict[str, Any]] = []
    for match in matches:
        indexed.append({
            "match": match,
            "home_norm": _norm_text(match["home_team"]),
            "away_norm": _norm_text(match["away_team"]),
        })

    rows: list[dict[str, Any]] = []
    for fixture in d.itertuples(index=False):
        home_norm = _norm_text(fixture.home_team)
        away_norm = _norm_text(fixture.away_team)
        kickoff = fixture.kickoff_utc
        candidates: list[dict[str, Any]] = []
        for item in indexed:
            match = item["match"]
            if item["home_norm"] != home_norm or item["away_norm"] != away_norm:
                continue
            delta_hours = abs(float((match["kickoff_utc"] - kickoff).total_seconds()) / 3600.0)
            if delta_hours <= delta_limit:
                candidates.append({"match": match, "delta_hours": delta_hours})
        candidates.sort(key=lambda x: (x["delta_hours"], x["match"]["match_id"]))
        if not candidates:
            rows.append({
                "match_id": str(fixture.match_id),
                "label_status": "EVENT_NOT_MATCHED",
                "event_id": "",
                "player_provider_id": "",
                "player_name": "",
                "player_team_name": "",
                "label_source": "fotmob_match_details_player_of_the_match",
                "label_retrieved_at_utc": "",
                "event_kickoff_utc": "",
                "event_match_delta_hours": None,
                "source_content_sha256": "",
                "league_retrieved_at_utc": league_retrieved_at,
            })
            continue
        best = candidates[0]
        if len(candidates) > 1 and abs(candidates[1]["delta_hours"] - best["delta_hours"]) < 1e-9:
            rows.append({
                "match_id": str(fixture.match_id),
                "label_status": "AMBIGUOUS_EVENT_MATCH",
                "event_id": "",
                "player_provider_id": "",
                "player_name": "",
                "player_team_name": "",
                "label_source": "fotmob_match_details_player_of_the_match",
                "label_retrieved_at_utc": "",
                "event_kickoff_utc": best["match"]["kickoff_utc"].isoformat(),
                "event_match_delta_hours": float(best["delta_hours"]),
                "source_content_sha256": "",
                "league_retrieved_at_utc": league_retrieved_at,
            })
            continue

        label = fetch_fotmob_match_potm(best["match"]["match_id"], fetcher=fetcher)
        if not label["cache_hit"] and delay > 0:
            import time
            time.sleep(delay)
        rows.append({
            "match_id": str(fixture.match_id),
            "label_status": "LABEL_FOUND" if label["label_found"] else "LABEL_MISSING",
            "event_id": label["event_id"],
            "player_provider_id": label["player_provider_id"],
            "player_name": label["player_name"],
            "player_team_name": label["player_team_name"],
            "label_source": label["source"],
            "label_retrieved_at_utc": label["retrieved_at_utc"],
            "event_kickoff_utc": best["match"]["kickoff_utc"].isoformat(),
            "event_match_delta_hours": float(best["delta_hours"]),
            "source_content_sha256": label["source_content_sha256"],
            "league_retrieved_at_utc": league_retrieved_at,
        })

    out = pd.DataFrame(rows)
    if not out.empty and out["match_id"].duplicated().any():
        raise RuntimeError("FotMob MOM label acquisition produced duplicate match_id rows")
    return out


def _player_name_keys(value: Any) -> set[str]:
    """Build conservative exact-name aliases for provider/display-name drift.

    Besides the full normalized name, support the common dataset convention
    "F. Surname" while retaining all surname tokens for compound surnames.
    No fuzzy distance is used here.
    """
    raw = str(value or "").strip()
    if not raw:
        return set()
    tokens = [t for t in raw.split() if t]
    out = {_norm_text(raw)}
    if len(tokens) >= 2:
        first = _norm_text(tokens[0])
        initial = first[:1]
        if initial:
            for suffix_len in range(1, min(3, len(tokens) - 1) + 1):
                surname = "".join(_norm_text(t) for t in tokens[-suffix_len:])
                if surname:
                    out.add(initial + surname)
    return {x for x in out if x}


def reconcile_fotmob_player_ids(
    labels: pd.DataFrame,
    feature_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    required_labels = {"match_id", "player_name", "label_status"}
    required_features = {"match_id", "player_id", "player_name"}
    missing_labels = sorted(required_labels - set(labels.columns))
    missing_features = sorted(required_features - set(feature_rows.columns))
    if missing_labels:
        raise ValueError(f"FotMob labels missing reconciliation columns: {missing_labels}")
    if missing_features:
        raise ValueError(f"MOM features missing reconciliation columns: {missing_features}")

    out = labels.copy()
    candidate_map: dict[tuple[str, str], set[str]] = {}
    global_name_map: dict[str, set[str]] = {}
    match_player_ids: dict[str, set[str]] = {}

    # Keep both a match-local map and a global exact-name map. The global map is
    # only a fallback: the resolved player must also exist in the target match.
    # This permits provider/display-name drift across matches without fuzzy guesses.
    for row in feature_rows.itertuples(index=False):
        match_id = str(row.match_id)
        player_id = str(row.player_id)
        match_player_ids.setdefault(match_id, set()).add(player_id)
        for name_key in _player_name_keys(row.player_name):
            candidate_map.setdefault((match_id, name_key), set()).add(player_id)
            global_name_map.setdefault(name_key, set()).add(player_id)

    resolved = 0
    missing = 0
    ambiguous = 0
    out["player_id"] = ""

    # Pass 1: original strict same-match name/alias resolution.
    for idx, row in out.iterrows():
        if str(row["label_status"]) != "LABEL_FOUND":
            continue
        candidates: set[str] = set()
        for name_key in _player_name_keys(row["player_name"]):
            candidates.update(candidate_map.get((str(row["match_id"]), name_key), set()))
        if len(candidates) == 1:
            out.at[idx, "player_id"] = next(iter(candidates))
            resolved += 1
        elif len(candidates) > 1:
            out.at[idx, "label_status"] = "AMBIGUOUS_PLAYER_NAME"
            ambiguous += 1

    # Build provider-ID mappings only from already-resolved labels. This avoids
    # allowing an unresolved label to teach the system its own identity.
    provider_to_player: dict[str, set[str]] = {}
    for row in out.itertuples(index=False):
        if str(row.label_status) != "LABEL_FOUND":
            continue
        provider_id = str(getattr(row, "player_provider_id", "") or "").strip()
        player_id = str(getattr(row, "player_id", "") or "").strip()
        if provider_id and player_id:
            provider_to_player.setdefault(provider_id, set()).add(player_id)

    # Pass 2: a global provider-ID or globally unique exact-name/alias fallback
    # is accepted only when the candidate player is present in that target match.
    for idx, row in out.iterrows():
        if str(row["label_status"]) != "LABEL_FOUND" or str(row.get("player_id", "")):
            continue
        match_id = str(row["match_id"])
        allowed_ids = match_player_ids.get(match_id, set())
        provider_id = str(row.get("player_provider_id", "") or "").strip()

        provider_candidates = {
            pid for pid in provider_to_player.get(provider_id, set())
            if pid in allowed_ids
        }
        if len(provider_candidates) == 1:
            out.at[idx, "player_id"] = next(iter(provider_candidates))
            resolved += 1
            continue
        if len(provider_candidates) > 1:
            out.at[idx, "label_status"] = "AMBIGUOUS_PLAYER_NAME"
            ambiguous += 1
            continue

        name_candidates: set[str] = set()
        for name_key in _player_name_keys(row["player_name"]):
            name_candidates.update(global_name_map.get(name_key, set()))
        name_candidates &= allowed_ids
        if len(name_candidates) == 1:
            out.at[idx, "player_id"] = next(iter(name_candidates))
            resolved += 1
        elif len(name_candidates) > 1:
            out.at[idx, "label_status"] = "AMBIGUOUS_PLAYER_NAME"
            ambiguous += 1
        else:
            out.at[idx, "label_status"] = "PLAYER_NOT_RECONCILED"
            missing += 1

    return out, {
        "resolved": resolved,
        "not_reconciled": missing,
        "ambiguous": ambiguous,
    }
