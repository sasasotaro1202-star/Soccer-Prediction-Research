import pandas as pd

from src.data import global_datalake_adapter as adapter


def test_global_datalake_adapter_maps_entities_and_preserves_pit(monkeypatch, tmp_path):
    fixtures = pd.DataFrame([
        {
            "id": 10,
            "date_utc": "2025-05-10T15:00:00Z",
            "league_id": 1,
            "home_team_id": 100,
            "away_team_id": 101,
            "goals_home": 2,
            "goals_away": 1,
            "home_xg": 1.8,
            "away_xg": 0.7,
            "home_possession": 57,
            "away_possession": 43,
            "home_fouls": 8,
            "away_fouls": 12,
            "home_offsides": 2,
            "away_offsides": 1,
            "home_pass_accuracy": 82,
            "away_pass_accuracy": 76,
            "home_goals_ht": 1,
            "away_goals_ht": 0,
            "home_xg_ht": 0.9,
            "away_xg_ht": 0.3,
            "known_at": "2025-05-10T16:45:00Z",
        },
        {
            "id": 11,
            "date_utc": "2025-05-11T15:00:00Z",
            "league_id": 99,
            "home_team_id": 100,
            "away_team_id": 101,
            "goals_home": 3,
            "goals_away": 3,
            "home_xg": 9.0,
            "away_xg": 9.0,
            "known_at": "2025-05-11T16:45:00Z",
        },
    ])
    leagues = pd.DataFrame([
        {"id": 1, "name": "Premier League"},
        {"id": 99, "name": "Unknown Global League"},
    ])
    teams = pd.DataFrame([
        {"id": 100, "name": "Alpha FC"},
        {"id": 101, "name": "Beta FC"},
    ])

    monkeypatch.setattr(adapter, "_download", lambda url, path: "a" * 64)

    def fake_read_parquet(path, engine=None):
        name = str(path)
        if name.endswith("fixtures.parquet"):
            return fixtures.copy()
        if name.endswith("leagues.parquet"):
            return leagues.copy()
        if name.endswith("teams.parquet"):
            return teams.copy()
        raise AssertionError(name)

    monkeypatch.setattr(adapter.pd, "read_parquet", fake_read_parquet)

    result, status = adapter.load_global_datalake_history(
        start_year=2025,
        end_year=2025,
        cache_dir=str(tmp_path),
    )

    assert status["status"] == "OK"
    assert len(result) == 1
    row = result.iloc[0]
    assert row["competition"] == "EPL"
    assert row["home_team"] == "Alpha FC"
    assert row["away_team"] == "Beta FC"
    assert row["home_goals"] == 2
    assert row["away_goals"] == 1
    assert row["source_available_at_utc"] == pd.Timestamp("2025-05-10T16:45:00Z")
    assert bool(row["pit_verified"])
    assert row["home_xg"] == 1.8
    assert row["away_xg"] == 0.7
    assert row["home_pass_accuracy"] == 82
    assert row["away_offsides"] == 1
