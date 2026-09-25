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
# API-Football league id for the English Premier League; do not confuse this with
# the dataset-internal league id (which is 1 in the pinned dataset snapshot).
DEFAULT_DATASET_API_FOOTBALL_LEAGUE_ID = 39
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
            WHERE l.api_football_id = {DEFAULT_DATASET_API_FOOTBALL_LEAGUE_ID}
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