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
    collect_sofascore_mom_labels_tournament_season,
    fetch_unique_tournament_seasons,
    select_season_id,
    label_data_contract_report,
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


def _load_frames(start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("duckdb is required for real-data MOM research; install it in the research environment") from exc

    urls = soccer_dataset_pinned_urls()
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
                at.name AS away_team
            FROM read_parquet('{urls["fixtures"]}') f
            JOIN read_parquet('https://huggingface.co/datasets/{SOCCER_DATASET_REPOSITORY.split("/",1)[1]}/resolve/{SOCCER_DATASET_COMMIT}/leagues.parquet') l
              ON f.league_id = l.id
            JOIN read_parquet('https://huggingface.co/datasets/{SOCCER_DATASET_REPOSITORY.split("/",1)[1]}/resolve/{SOCCER_DATASET_COMMIT}/teams.parquet') ht
              ON f.home_team_id = ht.id
            JOIN read_parquet('https://huggingface.co/datasets/{SOCCER_DATASET_REPOSITORY.split("/",1)[1]}/resolve/{SOCCER_DATASET_COMMIT}/teams.parquet') at
              ON f.away_team_id = at.id
            WHERE l.id = 39
              AND f.is_played = TRUE
              AND CAST(f.date_utc AS TIMESTAMP) >= TIMESTAMP '{start}'
              AND CAST(f.date_utc AS TIMESTAMP) < TIMESTAMP '{end}'
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
            SELECT fixture_id, MIN(known_at) AS known_at
            FROM read_parquet('{urls["match_stats"]}')
            WHERE known_at IS NOT NULL
            GROUP BY fixture_id
        """
        match_stats = con.execute(match_stats_sql).fetchdf()
        return fixtures, player_matches, player_stats, match_stats
    finally:
        con.close()


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
            "start": start,
            "end": end,
        },
    }

    fixtures, player_matches, player_stats, match_stats = _load_frames(start, end)
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

    label_fetcher = ExternalFetcher(cache_dir=str(root / "cache"), retries=3)
    seasons, season_retrieved_at = fetch_unique_tournament_seasons(
        tournament_id,
        fetcher=label_fetcher,
    )
    season_id = select_season_id(seasons, season_start_year)
    report["target"]["season_id"] = int(season_id)
    report["label_source"] = {
        "source": "sofascore_best_players_summary",
        "season_index_retrieved_at": season_retrieved_at,
    }

    labels = collect_sofascore_mom_labels_tournament_season(
        fixtures[["id", "date_utc", "home_team", "away_team"]].rename(columns={"id": "match_id", "date_utc": "kickoff_utc"}),
        tournament_id=tournament_id,
        season_id=season_id,
        cache_dir=str(root / "cache"),
        retries=3,
        max_event_pages=80,
    )
    labels.to_csv(root / "mom_labels.csv", index=False)
    label_report = label_data_contract_report(labels)
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
    parser.add_argument("--output-dir", default="artifacts/mom_research")
    args = parser.parse_args()
    return run(
        start=args.start,
        end=args.end,
        season_start_year=args.season_start_year,
        tournament_id=args.tournament_id,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
