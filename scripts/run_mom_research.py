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
    split_mom_development_locked,
)


DEFAULT_TOURNAMENT_ID = 17  # Premier League
DEFAULT_SEASON_START_YEAR = 2024
DEFAULT_START = "2024-08-01"
DEFAULT_END = "2025-06-01"
# Verified public SofaScore season identifier for Premier League 24/25.
# Keep this explicit for PIT/reproducibility; do not discover it from a challenge-prone endpoint.
DEFAULT_SOFASCORE_SEASON_ID = 61627


def _load_frames(
    start: str,
    end: str,
    *,
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
            WHERE l.api_football_id = 39
              AND lower(l.name) = 'premier league'
              AND lower(l.country) = 'england'
              AND f.is_played = TRUE
              AND CAST(f.date_utc AS TIMESTAMP) >= TIMESTAMP '{history_start_ts.strftime("%Y-%m-%d %H:%M:%S")}'
              AND CAST(f.date_utc AS TIMESTAMP) < TIMESTAMP '{target_end.strftime("%Y-%m-%d %H:%M:%S")}'
            ORDER BY f.date_utc, f.id
        """
        fixtures = con.execute(fixture_sql).fetchdf()
        if fixtures.empty:
            raise RuntimeError("No EPL fixtures found in the pinned dataset window")

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
    tournament_id: int = DEFAULT_TOURNAMENT_ID,
    season_id: int = DEFAULT_SOFASCORE_SEASON_ID,
    output_dir: str = "artifacts/mom_research",
) -> int:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "status": "STARTING",
        "production_adopted": False,
        "dataset": {
            "repository": SOCCER_DATASET_REPOSITORY,
            "commit": SOCCER_DATASET_COMMIT,
            "license": SOCCER_DATASET_LICENSE,
        },
        "target": {
            "tournament_id": int(tournament_id),
            "season_start_year": int(season_start_year),
            "season_id": int(season_id),
            "start": start,
            "end": end,
            "history_start": None,
        },
    }

    history_start = (pd.Timestamp(start) - pd.Timedelta(days=370)).strftime("%Y-%m-%d")
    report["target"]["history_start"] = history_start
    fixtures, player_matches, player_stats, match_stats = _load_frames(
        start,
        end,
        history_start=history_start,
    )
    target_fixtures = fixtures.loc[
        (pd.to_datetime(fixtures["date_utc"], utc=True) >= pd.Timestamp(start, tz="UTC"))
        & (pd.to_datetime(fixtures["date_utc"], utc=True) < pd.Timestamp(end, tz="UTC"))
    ].copy()
    if target_fixtures.empty:
        raise RuntimeError("No target EPL fixtures found after history expansion")
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
            report["label_source"] = {
                "source": "fotmob_match_details_player_of_the_match",
                "research_only_proxy": False,
                "production_safe": False,
                "league_id": int(DEFAULT_FOTMOB_LEAGUE_ID),
                "season": str(DEFAULT_FOTMOB_SEASON),
                "matching": "team_identity_and_kickoff_proximity",
                "player_mapping": "exact_normalized_name_with_single_candidate_required",
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
    report["models"] = {
        "binary_logit": binary_summary,
        "conditional_logit": conditional_summary,
    }

    # Development-only selection. Locked blocks are intentionally not inspected
    # for this choice; they remain a post-selection generalization check.
    bll = binary_summary["development_mean"]["logloss"]
    cll = conditional_summary["development_mean"]["logloss"]
    if cll + 1e-9 < bll:
        selected = "conditional_logit"
    else:
        selected = "binary_logit"
    report["development_selection"] = {
        "selected_method": selected,
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
    parser.add_argument("--tournament-id", type=int, default=DEFAULT_TOURNAMENT_ID)
    parser.add_argument("--season-id", type=int, default=DEFAULT_SOFASCORE_SEASON_ID)
    parser.add_argument("--output-dir", default="artifacts/mom_research")
    args = parser.parse_args()
    return run(
        start=args.start,
        end=args.end,
        season_start_year=args.season_start_year,
        tournament_id=args.tournament_id,
        season_id=args.season_id,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
