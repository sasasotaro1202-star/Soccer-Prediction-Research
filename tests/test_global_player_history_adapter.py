import pandas as pd

from src.data import global_player_history_adapter as adapter


def test_global_player_history_aggregates_prior_player_performance_with_match_stats_known_at(monkeypatch, tmp_path):
    fixtures = pd.DataFrame([
        {
            "id": 1,
            "date_utc": "2025-05-10T15:00:00Z",
            "league_id": 10,
            "home_team_id": 100,
            "away_team_id": 101,
        }
    ])
    leagues = pd.DataFrame([{"id": 10, "name": "Premier League"}])
    teams = pd.DataFrame([
        {"id": 100, "name": "Alpha FC"},
        {"id": 101, "name": "Beta FC"},
    ])
    match_stats = pd.DataFrame([
        {"fixture_id": 1, "known_at": "2025-05-10T16:45:00Z"},
    ])

    fixture_players = pd.DataFrame([
        {"id": 11, "fixture_id": 1, "team_id": 100, "player_id": 1001, "is_starter": True, "minutes": 90, "rating": 8.0},
        {"id": 12, "fixture_id": 1, "team_id": 100, "player_id": 1002, "is_starter": True, "minutes": 70, "rating": 7.0},
        {"id": 13, "fixture_id": 1, "team_id": 101, "player_id": 2001, "is_starter": True, "minutes": 90, "rating": 6.0},
    ])
    player_stats = pd.DataFrame([
        {
            "fixture_player_id": 11, "fixture_id": 1, "player_id": 1001,
            "games_rating": "8.0", "games_minutes": 90, "goals_assists": 2,
            "shots_total": 4, "shots_on": 2, "passes_key": 3, "duels_won": 5,
            "tackles_total": 2, "cards_yellow": 0, "cards_red": 0,
            "penalty_scored": 0, "penalty_missed": 0,
        },
        {
            "fixture_player_id": 12, "fixture_id": 1, "player_id": 1002,
            "games_rating": "7.0", "games_minutes": 70, "goals_assists": 1,
            "shots_total": 2, "shots_on": 1, "passes_key": 2, "duels_won": 3,
            "tackles_total": 4, "cards_yellow": 1, "cards_red": 0,
            "penalty_scored": 0, "penalty_missed": 0,
        },
        {
            "fixture_player_id": 13, "fixture_id": 1, "player_id": 2001,
            "games_rating": "6.0", "games_minutes": 90, "goals_assists": 0,
            "shots_total": 1, "shots_on": 0, "passes_key": 1, "duels_won": 2,
            "tackles_total": 5, "cards_yellow": 0, "cards_red": 0,
            "penalty_scored": 0, "penalty_missed": 0,
        },
    ])

    monkeypatch.setattr(adapter, "_download", lambda url, path: "a" * 64)

    def fake_read_parquet(path, columns=None, engine=None):
        name = str(path)
        if name.endswith("fixtures.parquet"):
            return fixtures.copy()
        if name.endswith("leagues.parquet"):
            return leagues.copy()
        if name.endswith("teams.parquet"):
            return teams.copy()
        if name.endswith("match_stats.parquet"):
            return match_stats.copy()
        raise AssertionError(name)

    monkeypatch.setattr(adapter.pd, "read_parquet", fake_read_parquet)

    def fake_batches(path, columns, batch_size=100_000):
        name = str(path)
        if name.endswith("fixture_players.parquet"):
            yield fixture_players.copy()
        elif name.endswith("fixture_players_stats_flat.parquet"):
            yield player_stats.copy()
        else:
            raise AssertionError(name)

    monkeypatch.setattr(adapter, "_read_parquet_batches", fake_batches)

    result, status = adapter.load_global_player_history(
        start_year=2025,
        end_year=2025,
        cache_dir=str(tmp_path),
    )

    assert status["status"] == "OK"
    assert status["pit_basis"] == "match_stats.known_at"
    assert status["production_status"] == "RESEARCH_ONLY"
    assert len(result) == 1

    row = result.iloc[0]
    assert row["home_team"] == "Alpha FC"
    assert row["away_team"] == "Beta FC"
    assert row["source_available_at_utc"] == pd.Timestamp("2025-05-10T16:45:00Z")
    assert bool(row["pit_verified"])
    assert row["home_player_lineup_size"] == 2
    assert row["home_player_starter_count"] == 2
    assert row["home_player_minutes_total"] == 160
    assert row["home_player_rating_mean"] == 7.5
    assert row["home_player_goals_assists_total"] == 3
    assert row["home_player_shots_total"] == 6
    assert row["home_player_passes_key_total"] == 5
    assert row["home_player_tackles_total"] == 6
    assert row["away_player_goals_assists_total"] == 0


def test_player_feature_columns_are_consumed_by_pit_feature_replay():
    from src.features.soccer_features import build_match_features

    history = pd.DataFrame([
        {
            "match_id": "prior-1",
            "competition": "EPL",
            "kickoff_utc": "2025-05-01T15:00:00Z",
            "home_team": "Alpha FC",
            "away_team": "Beta FC",
            "home_goals": 2,
            "away_goals": 1,
            "source_available_at_utc": "2025-05-01T16:45:00Z",
            "home_player_rating_mean": 7.5,
            "away_player_rating_mean": 6.0,
            "home_player_starter_rating_mean": 7.8,
            "away_player_starter_rating_mean": 6.2,
            "home_player_goals_assists_total": 3,
            "away_player_goals_assists_total": 0,
            "home_player_shots_total": 6,
            "away_player_shots_total": 1,
            "home_player_tackles_total": 6,
            "away_player_tackles_total": 5,
            "home_player_lineup_size": 2,
            "away_player_lineup_size": 1,
            "home_player_starter_count": 2,
            "away_player_starter_count": 1,
        },
    ])
    future = pd.DataFrame([{
        "match_id": "future-1",
        "competition": "EPL",
        "kickoff_utc": "2025-05-05T15:00:00Z",
        "home_team": "Alpha FC",
        "away_team": "Beta FC",
        "neutral_venue": False,
    }])
    result = build_match_features(history, future, windows=(1,))
    row = result.iloc[0]
    assert row["home_player_rating_mean_1"] == 7.5
    assert row["away_player_rating_mean_1"] == 6.0
    assert row["player_goals_assists_total_diff_1"] == 3.0
    assert row["player_shots_total_diff_1"] == 5.0
    assert row["player_tackles_total_diff_1"] == 1.0
