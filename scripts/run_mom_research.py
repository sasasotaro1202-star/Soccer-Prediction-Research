"""Bounded real-data MOM research run.

Default target: Premier League 2024-25 (SofaScore tournament 17). The script uses
pinned public parquet data for pre-match player history and SofaScore only for the
post-match MOM label. It never promotes a model to production.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.competition_catalog import COMPETITION_CATALOG
from src.data.competition_sources import TARGET_COMPETITIONS
from src.data.mom_player_history_adapter import (
    SOCCER_DATASET_COMMIT,
    SOCCER_DATASET_LICENSE,
    SOCCER_DATASET_REPOSITORY,
    build_mom_feature_rows,
    attach_mom_labels,
    mom_data_contract_report,
    soccer_dataset_pinned_urls,
)
from src.data.sofascore_mom_labels import (
    collect_sofascore_mom_labels,
    collect_sofascore_mom_labels_tournament_season,
    fetch_unique_football_tournaments,
    resolve_unique_tournament_id,
    discover_unique_tournament_from_scheduled_events,
    discover_unique_season_from_scheduled_events,
    fetch_unique_tournament_seasons,
    select_season_id,
    label_data_contract_report,
)
from src.data.fotmob_mom_labels import (
    DEFAULT_FOTMOB_LEAGUE_ID,
    DEFAULT_FOTMOB_SEASON,
    collect_fotmob_mom_labels,
    reconcile_fotmob_player_ids,
)
from src.data.external_fetch import ExternalFetcher
from src.evaluation.mom_walk_forward import (
    run_mom_walk_forward,
    run_mom_walk_forward_calibrated_soft_ensemble,
    split_mom_development_locked,
)


DEFAULT_TOURNAMENT_ID = 17  # Premier League
DEFAULT_SEASON_START_YEAR = 2024
DEFAULT_START = "2024-08-01"
DEFAULT_END = "2025-06-01"
# Verified public SofaScore season identifier for Premier League 24/25.
# Keep this explicit for PIT/reproducibility; do not discover it from a challenge-prone endpoint.
DEFAULT_SOFASCORE_SEASON_ID = 61627

# Stable SofaScore unique-tournament IDs are used before registry discovery.
# These public URL identifiers are versioned here so a transient discovery
# endpoint failure cannot block recurring competitions. Season IDs remain
# dynamic and are still resolved through the tournament seasons endpoint.
SOFASCORE_STATIC_TOURNAMENT_IDS = {
    "EPL": 17, "ERE": 37, "LL": 8, "SA": 23, "BL1": 35, "FL1": 34,
    "J1": 196, "J2": 402, "UCL": 7, "UEL": 679, "UECL": 17015,
    "UEFA_NATIONS_LEAGUE_M": 10783,
    "UEFA_SUPER_CUP": 465, "UEFA_YOUTH_LEAGUE": 2132, "UWCL": 696,
}

DATASET_NAME_ALIASES = {
    "EPL": ("Premier League",), "ERE": ("Eredivisie",), "LL": ("LaLiga", "La Liga"),
    "SA": ("Serie A",), "BL1": ("Bundesliga",), "FL1": ("Ligue 1",),
    "J1": ("J1 League", "J.League"), "J2": ("J2 League", "J.League 2"), "J3": ("J3 League", "J.League 3"),
    "UCL": ("UEFA Champions League", "Champions League"), "UEL": ("UEFA Europa League", "Europa League"), "UECL": ("UEFA Conference League", "Conference League"),
    "UEFA_SUPER_CUP": ("UEFA Super Cup",), "UEFA_YOUTH_LEAGUE": ("UEFA Youth League",),
    "UWCL": ("UEFA Women's Champions League", "Women's Champions League"), "UWEC": ("UEFA Women's Europa Cup", "Women's Europa League"),
    "UEFA_EURO_M": ("European Championship", "UEFA European Championship"), "UEFA_EURO_QUALI_M": ("European Championship Qualification", "UEFA European Qualifiers"),
    "UEFA_NATIONS_LEAGUE_M": ("UEFA Nations League",), "UEFA_EURO_W": ("Women's European Championship", "UEFA Women's European Championship"),
    "UEFA_EURO_QUALI_W": ("Women's European Championship Qualification", "UEFA Women's European Qualifiers"), "UEFA_NATIONS_LEAGUE_W": ("UEFA Women's Nations League", "Women's Nations League"),
    "UEFA_U21": ("UEFA European Under-21 Championship", "European U-21 Championship"), "UEFA_U19": ("UEFA European Under-19 Championship",), "UEFA_U17": ("UEFA European Under-17 Championship",),
    "UEFA_WU19": ("UEFA Women's Under-19 Championship",), "UEFA_WU17": ("UEFA Women's Under-17 Championship",), "UEFA_REGIONS_CUP": ("UEFA Regions' Cup",),
    "EMP_CUP": ("Emperor's Cup", "Japan FA Cup"), "INTL_M": ("Friendlies", "International Friendlies", "International Friendly"), "INTL_W": ("Women's Friendlies", "International Friendlies Women"),
    "U23_M": ("International U23", "U-23"), "U18_M": ("International U18", "U-18"), "AG_M": ("Asian Games", "Asian Games Men"), "AG_W": ("Asian Games", "Asian Games Women"),
}


def _competition_spec(code: str):
    code = str(code).strip()
    if code not in TARGET_COMPETITIONS:
        raise ValueError(f"Unsupported active competition code: {code!r}")
    matches = [x for x in COMPETITION_CATALOG if x.code == code]
    if len(matches) != 1:
        raise RuntimeError(f"Competition catalog is inconsistent for {code!r}")
    return matches[0]


def _resolve_dataset_league_ids(con, dataset_base: str, competition_code: str) -> list[int]:
    spec = _competition_spec(competition_code)
    names = list(dict.fromkeys((spec.name, *DATASET_NAME_ALIASES.get(competition_code, ()))))
    values = ", ".join("'" + str(x).replace("'", "''").lower() + "'" for x in names)
    rows = con.execute(f"SELECT id, name, country FROM read_parquet('{dataset_base}/leagues.parquet') WHERE lower(name) IN ({values}) ORDER BY id").fetchdf()
    if rows.empty:
        return []
    records = rows.to_dict(orient="records")
    region = str(spec.region).strip().lower()
    preferred = [x for x in records if str(x.get("country") or "").strip().lower() == region]
    selected = preferred if preferred else records
    return [int(x) for x in pd.to_numeric(pd.Series([x["id"] for x in selected]), errors="coerce").dropna().astype(int)]


def _resolve_sofascore_scope(
    competition_code: str,
    season_start_year: int,
    *,
    tournament_id: int | None,
    season_id: int | None,
    fetcher: ExternalFetcher,
    discovery_dates_utc: list[str] | None = None,
) -> tuple[int, int, dict[str, Any]]:
    spec = _competition_spec(competition_code)
    names = [str(spec.name), *DATASET_NAME_ALIASES.get(competition_code, ())]
    category_names = [str(spec.region), str(spec.competition_type)]
    meta: dict[str, Any] = {
        "competition": competition_code,
        "tournament_discovery": "DYNAMIC_EXACT_NAME",
    }
    if tournament_id is None:
        static_id = SOFASCORE_STATIC_TOURNAMENT_IDS.get(str(competition_code))
        if static_id is not None:
            tournament_id = int(static_id)
            meta["tournament_discovery"] = "STATIC_VERSIONED_PUBLIC_ID"
            meta["tournament_id_source_url"] = f"https://www.sofascore.com/football/tournament/{int(tournament_id)}"
        else:
            registry_error: Exception | None = None
            try:
                tournaments, retrieved_at = fetch_unique_football_tournaments(fetcher=fetcher)
                tournament_id = resolve_unique_tournament_id(
                    tournaments,
                    names=names,
                    category_names=category_names,
                )
                meta["tournament_registry_retrieved_at_utc"] = retrieved_at
            except (RuntimeError, ValueError) as exc:
                registry_error = exc
                if not discovery_dates_utc:
                    raise RuntimeError(
                        "Registry tournament discovery failed and event-based fallback has no fixture dates: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                tournament_id, discovery_meta = discover_unique_tournament_from_scheduled_events(
                    discovery_dates_utc,
                    names=names,
                    category_names=category_names,
                    fetcher=fetcher,
                    max_dates=5,
                )
                meta["tournament_discovery"] = "DYNAMIC_EVENT_UNIQUE_TOURNAMENT_FALLBACK"
                meta["registry_fallback_reason"] = f"{type(registry_error).__name__}: {registry_error}"
                meta["event_discovery"] = discovery_meta
    meta["tournament_id"] = int(tournament_id)
    if season_id is None:
        try:
            seasons, retrieved_at = fetch_unique_tournament_seasons(
                int(tournament_id), fetcher=fetcher
            )
            season_id = select_season_id(seasons, int(season_start_year))
            meta["season_registry_retrieved_at_utc"] = retrieved_at
        except (RuntimeError, ValueError) as registry_exc:
            if not discovery_dates_utc:
                raise RuntimeError(
                    "Registry season discovery failed and event-based fallback has no fixture dates: "
                    f"{type(registry_exc).__name__}: {registry_exc}"
                ) from registry_exc
            season_id, season_meta = discover_unique_season_from_scheduled_events(
                discovery_dates_utc,
                tournament_id=int(tournament_id),
                season_start_year=int(season_start_year),
                fetcher=fetcher,
                max_dates=5,
            )
            meta["season_discovery"] = season_meta
            meta["season_registry_fallback_reason"] = (
                f"{type(registry_exc).__name__}: {registry_exc}"
            )
    meta["season_id"] = int(season_id)
    return int(tournament_id), int(season_id), meta


def _load_frames(
    start: str,
    end: str,
    *,
    competition_code: str = "EPL",
    history_start: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("duckdb is required for real-data MOM research; install it in the research environment") from exc

    urls = soccer_dataset_pinned_urls()
    owner, name = SOCCER_DATASET_REPOSITORY.split("/", 1)
    dataset_base = f"https://huggingface.co/datasets/{owner}/{name}/resolve/{SOCCER_DATASET_COMMIT}"
    target_start = pd.Timestamp(start)
    target_end = pd.Timestamp(end)
    if target_end <= target_start:
        raise ValueError("MOM target end must be after target start")
    if history_start is None:
        history_start = (target_start - pd.Timedelta(days=370)).strftime("%Y-%m-%d")
    history_start_ts = pd.Timestamp(history_start)
    if history_start_ts >= target_start:
        raise ValueError("MOM history_start must precede target start")
    con = duckdb.connect()
    try:
        try:
            con.execute("INSTALL httpfs")
            con.execute("LOAD httpfs")
        except Exception:
            # Modern DuckDB builds may autoload httpfs, so retrying the query
            # below is the authoritative check.
            pass

        league_ids = _resolve_dataset_league_ids(con, dataset_base, competition_code)
        if not league_ids:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        league_ids_sql = ", ".join(str(x) for x in league_ids)

        fixture_sql = f"""
            SELECT
                f.id,
                f.date_utc,
                f.home_team_id,
                f.away_team_id,
                f.goals_home,
                f.goals_away,
                f.is_played,
                ht.name AS home_team,
                away_team_ref.name AS away_team
            FROM read_parquet('{urls["fixtures"]}') f
            JOIN read_parquet('{dataset_base}/leagues.parquet') l
              ON f.league_id = l.id
            JOIN read_parquet('{dataset_base}/teams.parquet') ht
              ON f.home_team_id = ht.id
            JOIN read_parquet('{dataset_base}/teams.parquet') away_team_ref
              ON f.away_team_id = away_team_ref.id
            WHERE f.league_id IN ({league_ids_sql})
              AND f.is_played = TRUE
              AND CAST(f.date_utc AS TIMESTAMP) >= TIMESTAMP '{history_start_ts.strftime("%Y-%m-%d %H:%M:%S")}'
              AND CAST(f.date_utc AS TIMESTAMP) < TIMESTAMP '{target_end.strftime("%Y-%m-%d %H:%M:%S")}'
            ORDER BY f.date_utc, f.id
        """
        fixtures = con.execute(fixture_sql).fetchdf()
        if fixtures.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        con.register("target_fixture_ids", fixtures[["id"]].rename(columns={"id": "fixture_id"}))

        player_sql = f"""
            SELECT
                p.fixture_id,
                p.team_id,
                p.player_id,
                p.player_name,
                p.is_starter,
                p.position,
                p.minutes,
                p.rating
            FROM read_parquet('{urls["fixture_players"]}') p
            JOIN target_fixture_ids f ON p.fixture_id = f.fixture_id
        """
        player_matches = con.execute(player_sql).fetchdf()

        stats_sql = f"""
            SELECT
                s.fixture_id,
                s.player_id,
                s.games_minutes,
                s.games_rating,
                s.goals_total,
                s.goals_assists,
                s.shots_total,
                s.passes_key
            FROM read_parquet('{urls["fixture_players_stats_flat"]}') s
            JOIN target_fixture_ids f
              ON s.fixture_id = f.fixture_id
        """
        player_stats = con.execute(stats_sql).fetchdf()

        match_stats_sql = f"""
            SELECT s.fixture_id, MIN(s.known_at) AS known_at
            FROM read_parquet('{urls["match_stats"]}') s
            JOIN target_fixture_ids f ON s.fixture_id = f.fixture_id
            WHERE s.known_at IS NOT NULL
            GROUP BY s.fixture_id
        """
        match_stats = con.execute(match_stats_sql).fetchdf()
        return fixtures, player_matches, player_stats, match_stats
    finally:
        con.close()


def _build_dataset_rating_proxy_labels(
    player_matches: pd.DataFrame,
    target_fixtures: pd.DataFrame,
    player_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a research-only post-match performance proxy label.

    Priority is deterministic:
    1) highest provider rating when available;
    2) otherwise goals, assists, key passes, shots, minutes.
    This label is never treated as an official MOM award and is never production-safe.
    """
    required = {"fixture_id", "player_id", "player_name"}
    missing = sorted(required - set(player_matches.columns))
    if missing:
        raise RuntimeError(f"Proxy MOM labels missing player columns: {missing}")

    p = player_matches.copy()
    for col in ("fixture_id", "player_id"):
        p[col] = pd.to_numeric(p[col], errors="coerce")
    for col in ("rating", "minutes"):
        if col not in p.columns:
            p[col] = np.nan
        p[col] = pd.to_numeric(p[col], errors="coerce")

    if player_stats is not None and {"fixture_id", "player_id"}.issubset(player_stats.columns):
        ps = player_stats.copy()
        for col in ("fixture_id", "player_id"):
            ps[col] = pd.to_numeric(ps[col], errors="coerce")
        select_cols = ["fixture_id", "player_id"]
        for col in ("games_rating", "games_minutes", "goals_total", "goals_assists", "passes_key", "shots_total"):
            if col in ps.columns:
                select_cols.append(col)
        ps = ps[select_cols].copy()
        if ps.duplicated(["fixture_id", "player_id"]).any():
            raise RuntimeError("Proxy MOM player stats contain duplicate fixture/player rows")
        p = p.merge(ps, on=["fixture_id", "player_id"], how="left", validate="one_to_one")
        if "games_rating" in p.columns:
            p["rating"] = p["rating"].combine_first(pd.to_numeric(p["games_rating"], errors="coerce"))
        if "games_minutes" in p.columns:
            p["minutes"] = p["minutes"].combine_first(pd.to_numeric(p["games_minutes"], errors="coerce"))

    for col in ("goals_total", "goals_assists", "passes_key", "shots_total"):
        if col not in p.columns:
            p[col] = np.nan
        p[col] = pd.to_numeric(p[col], errors="coerce")

    p = p.merge(
        target_fixtures[["id"]].rename(columns={"id": "fixture_id"}),
        on="fixture_id",
        how="inner",
        validate="many_to_one",
    )
    p = p.dropna(subset=["fixture_id", "player_id"]).copy()
    if p.empty:
        raise RuntimeError("No target player rows available for proxy MOM labels")

    rows: list[dict[str, Any]] = []
    for fixture_id, g in p.groupby("fixture_id", sort=True):
        has_rating = np.isfinite(g["rating"].to_numpy(dtype=float))
        if has_rating.any():
            g = g.loc[has_rating].sort_values(
                ["rating", "minutes", "player_id"],
                ascending=[False, False, True],
                kind="mergesort",
            )
            winner = g.iloc[0]
            source = "dataset_rating_top_performer_proxy"
        else:
            for col in ("goals_total", "goals_assists", "passes_key", "shots_total", "minutes"):
                g[col] = g[col].fillna(-np.inf)
            g = g.sort_values(
                ["goals_total", "goals_assists", "passes_key", "shots_total", "minutes", "player_id"],
                ascending=[False, False, False, False, False, True],
                kind="mergesort",
            )
            winner = g.iloc[0]
            source = "dataset_boxscore_top_performer_proxy"
        rows.append({
            "match_id": str(int(fixture_id)),
            "label_status": "LABEL_FOUND",
            "event_id": f"dataset-performance-proxy:{int(fixture_id)}",
            "player_id": str(int(winner["player_id"])),
            "player_name": str(winner["player_name"]),
            "label_source": source,
            "label_retrieved_at_utc": "",
            "event_kickoff_utc": "",
            "event_match_delta_hours": None,
            "source_content_sha256": "",
        })
    return pd.DataFrame(rows)

def _summarize_metrics(metrics: pd.DataFrame) -> dict[str, Any]:
    dev, locked = split_mom_development_locked(metrics, locked_blocks=2)
    dev_mean = {
        "top1_hit_rate": float(dev["top1_hit_rate"].mean()),
        "top4_hit_rate": float(dev["top4_hit_rate"].mean()),
        "mrr": float(dev["mrr"].mean()),
        "ndcg_at_4": float(dev["ndcg_at_4"].mean()),
        "logloss": float(dev["logloss"].mean()),
        "brier": float(dev["brier"].mean()),
        "ece": float(dev["ece"].mean()),
    }
    locked_mean = {
        "top1_hit_rate": float(locked["top1_hit_rate"].mean()),
        "top4_hit_rate": float(locked["top4_hit_rate"].mean()),
        "mrr": float(locked["mrr"].mean()),
        "ndcg_at_4": float(locked["ndcg_at_4"].mean()),
        "logloss": float(locked["logloss"].mean()),
        "brier": float(locked["brier"].mean()),
        "ece": float(locked["ece"].mean()),
    }
    return {
        "blocks": metrics.to_dict(orient="records"),
        "development_mean": dev_mean,
        "locked_mean": locked_mean,
        "development_blocks": dev["block"].astype(int).tolist(),
        "locked_blocks": locked["block"].astype(int).tolist(),
    }


def run(
    *,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    season_start_year: int = DEFAULT_SEASON_START_YEAR,
    competition_code: str = "EPL",
    tournament_id: int | None = None,
    season_id: int | None = None,
    output_dir: str = "artifacts/mom_research",
) -> int:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    spec = _competition_spec(competition_code)
    report: dict[str, Any] = {
        "status": "STARTING",
        "production_adopted": False,
        "competition": {"code": str(competition_code), "name": str(spec.name), "region": str(spec.region), "type": str(spec.competition_type)},
        "dataset": {
            "repository": SOCCER_DATASET_REPOSITORY,
            "commit": SOCCER_DATASET_COMMIT,
            "license": SOCCER_DATASET_LICENSE,
        },
        "target": {
            "tournament_id": None,
            "season_start_year": int(season_start_year),
            "season_id": None,
            "start": start,
            "end": end,
            "history_start": None,
        },
    }

    history_start = (pd.Timestamp(start) - pd.Timedelta(days=370)).strftime("%Y-%m-%d")
    report["target"]["history_start"] = history_start
    fetcher = ExternalFetcher(cache_dir=str(root / "cache"), retries=3)

    # Load the pinned fixture data before resolving the SofaScore tournament id.
    # This gives the event-based fallback deterministic real fixture dates.
    fixtures, player_matches, player_stats, match_stats = _load_frames(
        start,
        end,
        competition_code=competition_code,
        history_start=history_start,
    )
    if fixtures.empty:
        report["status"] = "DEFERRED_NO_COMPETITION_FIXTURES"
        (root / "mom_research_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return 0

    discovery_dates = (
        pd.to_datetime(fixtures["date_utc"], utc=True, errors="coerce")
        .dropna()
        .dt.strftime("%Y-%m-%d")
        .drop_duplicates()
        .tolist()
    )
    try:
        tournament_id, season_id, scope_meta = _resolve_sofascore_scope(
            competition_code,
            int(season_start_year),
            tournament_id=tournament_id,
            season_id=season_id,
            fetcher=fetcher,
            discovery_dates_utc=discovery_dates,
        )
    except (RuntimeError, ValueError) as exc:
        report["status"] = "DEFERRED_SCOPE_RESOLUTION"
        report["scope_resolution_error"] = f"{type(exc).__name__}: {exc}"
        (root / "mom_research_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return 0
    report["target"]["tournament_id"] = int(tournament_id)
    report["target"]["season_id"] = int(season_id)
    report["scope_resolution"] = scope_meta
    target_fixtures = fixtures.loc[
        (pd.to_datetime(fixtures["date_utc"], utc=True) >= pd.Timestamp(start, tz="UTC"))
        & (pd.to_datetime(fixtures["date_utc"], utc=True) < pd.Timestamp(end, tz="UTC"))
    ].copy()
    if target_fixtures.empty:
        report["status"] = "DEFERRED_NO_TARGET_FIXTURES"
        (root / "mom_research_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return 0
    features = build_mom_feature_rows(
        fixtures,
        player_matches,
        player_stats,
        match_stats,
        min_history_appearances=3,
        lookback_appearances=10,
    )
    feature_report = mom_data_contract_report(features)
    report["feature_contract"] = feature_report
    if not str(feature_report.get("status", "")).startswith("READY"):
        report["status"] = "DEFERRED_FEATURE_COVERAGE"
        (root / "mom_research_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return 0

    # Locate historical events directly through the pinned tournament/season
    # event index. This avoids the daily schedule endpoint's WAF/payload drift
    # and uses the already-resolved season_id for a deterministic historical scope.
    report["label_source"] = {
        "source": "sofascore_tournament_season_events_best_players_summary",
        "tournament_id": int(tournament_id),
        "season_id": int(season_id),
        "matching": "team_identity_and_kickoff_proximity",
        "pit_role": "outcome_only",
        "max_event_pages": 60,
    }

    label_fixtures = target_fixtures[["id", "date_utc", "home_team", "away_team"]].rename(
        columns={"id": "match_id", "date_utc": "kickoff_utc"}
    )
    try:
        labels = collect_sofascore_mom_labels_tournament_season(
            label_fixtures,
            tournament_id=int(tournament_id),
            season_id=int(season_id),
            cache_dir=str(root / "cache"),
            retries=3,
            max_event_pages=60,
        )
        report["label_source"] = {
            "source": "sofascore_tournament_season_events_best_players_summary",
            "research_only_proxy": False,
            "production_safe": False,
            "tournament_id": int(tournament_id),
            "season_id": int(season_id),
            "matching": "team_identity_and_kickoff_proximity",
            "pit_role": "outcome_only",
            "max_event_pages": 60,
        }
    except RuntimeError as sofa_exc:
        # Use a second independent public provider before falling back to a
        # post-match performance proxy. FotMob's matchDetails exposes
        # content.matchFacts.playerOfTheMatch, but its player ids are provider-
        # specific and must be reconciled against the PIT-safe candidate rows.
        try:
            if str(competition_code) != "EPL":
                raise RuntimeError("FotMob fallback is pinned to EPL; other competitions remain SofaScore/proxy only")
            labels = collect_fotmob_mom_labels(
                label_fixtures,
                league_id=DEFAULT_FOTMOB_LEAGUE_ID,
                season=DEFAULT_FOTMOB_SEASON,
                cache_dir=str(root / "cache"),
                retries=3,
                request_delay_seconds=0.25,
            )
            labels, reconciliation = reconcile_fotmob_player_ids(labels, features)
            report["player_reconciliation"] = reconciliation
            if "reconciliation_method" in labels.columns:
                report["player_reconciliation"]["methods"] = (
                    labels.loc[labels["label_status"].eq("LABEL_FOUND"), "reconciliation_method"]
                    .value_counts(dropna=False)
                    .to_dict()
                )
            report["label_source"] = {
                "source": "fotmob_match_details_player_of_the_match",
                "research_only_proxy": False,
                "production_safe": False,
                "league_id": int(DEFAULT_FOTMOB_LEAGUE_ID),
                "season": str(DEFAULT_FOTMOB_SEASON),
                "matching": "team_identity_and_kickoff_proximity",
                "player_mapping": "conservative_exact_name_aliases_with_unique_candidate_required",
                "pit_role": "outcome_only",
                "fallback_reason": f"{type(sofa_exc).__name__}: {sofa_exc}",
            }
        except RuntimeError as fotmob_exc:
            # External MOM labels are unavailable from both public providers.
            # Keep the explicit post-match performance proxy for architecture
            # research only; it is never production-safe.
            labels = _build_dataset_rating_proxy_labels(
                player_matches,
                target_fixtures,
                player_stats=player_stats,
            )
            report["label_source"] = {
                "source": "dataset_rating_top_performer_proxy",
                "research_only_proxy": True,
                "production_safe": False,
                "fallback_reason": (
                    f"sofascore={type(sofa_exc).__name__}: {sofa_exc}; "
                    f"fotmob={type(fotmob_exc).__name__}: {fotmob_exc}"
                ),
            }
    labels.to_csv(root / "mom_labels.csv", index=False)
    label_report = label_data_contract_report(labels)
    label_report["research_only_proxy"] = bool(
        report.get("label_source", {}).get("research_only_proxy", False)
    )
    report["label_contract"] = label_report

    # A post-match proxy is useful for architecture experiments, but it is not
    # a valid substitute for the actual MOM target. Never turn proxy labels into
    # apparent OOS evidence for a competition without a verified MOM label source.
    if bool(label_report["research_only_proxy"]):
        report["status"] = "DEFERRED_RESEARCH_ONLY_PROXY_LABELS"
        report["oos_evaluated"] = False
        report["defer_reason"] = "Actual post-match MOM labels unavailable from verified public providers."
        (root / "mom_research_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return 0

    if int(label_report.get("label_found", 0)) < 100:
        report["status"] = "DEFERRED_INSUFFICIENT_MOM_LABELS"
        (root / "mom_research_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return 0

    labels_for_join = labels.loc[
        labels["label_status"].eq("LABEL_FOUND"),
        ["match_id", "player_id"],
    ].copy()
    labelled = attach_mom_labels(features, labels_for_join)
    labelled.to_csv(root / "mom_labelled_features.csv", index=False)
    report["labelled_matches"] = int(labelled["match_id"].nunique())
    report["labelled_rows"] = int(len(labelled))
    if report["labelled_matches"] < 100:
        report["status"] = "DEFERRED_INSUFFICIENT_LABELED_CANDIDATE_MATCHES"
        (root / "mom_research_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return 0

    binary = run_mom_walk_forward(
        labelled,
        n_blocks=6,
        locked_blocks=2,
        min_train_matches=30,
        method="binary_logit",
    )
    conditional = run_mom_walk_forward(
        labelled,
        n_blocks=6,
        locked_blocks=2,
        min_train_matches=30,
        method="conditional_logit",
    )
    binary_summary = _summarize_metrics(binary)
    conditional_summary = _summarize_metrics(conditional)
    hist_gbdt = run_mom_walk_forward(
        labelled,
        n_blocks=6,
        locked_blocks=2,
        min_train_matches=30,
        method="hist_gbdt",
    )
    hist_gbdt_summary = _summarize_metrics(hist_gbdt)
    soft_ensemble = run_mom_walk_forward(
        labelled,
        n_blocks=6,
        locked_blocks=2,
        min_train_matches=30,
        method="soft_ensemble",
    )
    soft_ensemble_summary = _summarize_metrics(soft_ensemble)
    soft_ensemble_calibrated = run_mom_walk_forward_calibrated_soft_ensemble(
        labelled,
        n_blocks=6,
        locked_blocks=2,
        min_train_matches=30,
        min_calibration_matches=30,
        prior_strength=120.0,
    )
    soft_ensemble_calibrated_summary = _summarize_metrics(soft_ensemble_calibrated)
    report["models"] = {
        "binary_logit": binary_summary,
        "conditional_logit": conditional_summary,
        "hist_gbdt": hist_gbdt_summary,
        "soft_ensemble_equal_weight": soft_ensemble_summary,
        "soft_ensemble_temperature_calibrated": soft_ensemble_calibrated_summary,
    }

    # Development-only selection. Locked blocks are intentionally not inspected
    # for this choice; they remain a post-selection generalization check.
    model_summaries = {
        "binary_logit": binary_summary,
        "conditional_logit": conditional_summary,
        "hist_gbdt": hist_gbdt_summary,
        "soft_ensemble_equal_weight": soft_ensemble_summary,
        "soft_ensemble_temperature_calibrated": soft_ensemble_calibrated_summary,
    }
    # The operational output is exactly four players, so model selection must
    # optimize the ranking task itself rather than an unrelated per-row probability
    # objective. The rule is fixed before looking at locked blocks:
    #   1) maximize development Top-4 hit rate;
    #   2) break ties with MRR;
    #   3) break remaining ties with lower LogLoss.
    # Locked blocks remain score-only generalization evidence.
    selection_scores = {
        name: (
            float(summary["development_mean"]["top4_hit_rate"]),
            float(summary["development_mean"]["mrr"]),
            -float(summary["development_mean"]["logloss"]),
        )
        for name, summary in model_summaries.items()
    }
    selected = max(selection_scores, key=selection_scores.get)
    report["development_selection"] = {
        "selected_method": selected,
        "selection_objective": "development_top4_hit_rate_then_mrr_then_logloss",
        "selection_scores": selection_scores,
        "locked_blocks_untouched_for_selection": True,
    }
    report["status"] = "RESEARCH_EVALUATED"
    report["calibration_status"] = "NOT_YET_CALIBRATED"
    report["production_adopted"] = False
    (root / "mom_research_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--season-start-year", type=int, default=DEFAULT_SEASON_START_YEAR)
    parser.add_argument("--competition-code", default="EPL")
    parser.add_argument("--tournament-id", type=int, default=None)
    parser.add_argument("--season-id", type=int, default=None)
    parser.add_argument("--output-dir", default="artifacts/mom_research")
    args = parser.parse_args()
    return run(
        start=args.start,
        end=args.end,
        season_start_year=args.season_start_year,
        competition_code=args.competition_code,
        tournament_id=args.tournament_id,
        season_id=args.season_id,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
