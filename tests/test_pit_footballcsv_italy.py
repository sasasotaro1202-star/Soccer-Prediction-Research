import pandas as pd

import src.data.pit_footballcsv_italy as provider


def test_provider_accepts_only_exact_unique_result_after_lower_bound(monkeypatch):
    history = pd.DataFrame(
        {
            "competition": ["SA", "SA"],
            "season_start": [2020, 2020],
            "kickoff_utc": pd.to_datetime(["2020-09-19", "2020-09-20"], utc=True),
            "kickoff_time_available": [False, False],
            "home_team": ["Fiorentina", "Verona"],
            "away_team": ["Torino", "Roma"],
            "home_goals": [1, 0],
            "away_goals": [0, 0],
        },
        index=[20, 21],
    )
    commits = [
        {"sha": "too-early", "commit": {"committer": {"date": "2020-09-20T23:59:00Z"}}},
        {"sha": "valid", "commit": {"committer": {"date": "2020-09-21T00:00:00Z"}}},
    ]
    snapshot = """Date,Team 1,FT,HT,Team 2
Sat Sep 19 2020,Fiorentina,1-0,0-0,Torino
Sun Sep 20 2020,Verona,0-0,0-0,Roma
"""
    monkeypatch.setattr(provider, "_commits", lambda *args, **kwargs: commits)
    monkeypatch.setattr(provider, "_snapshot", lambda *args, **kwargs: snapshot)
    result = provider.apply_footballcsv_italy_pit(history, cache_dir="/tmp/unused")
    assert result.index.tolist() == [20, 21]
    assert all(result["pit_evidence_status"].eq("VERIFIED"))
    assert all(result["source_available_at_utc"].str.startswith("2020-09-21"))


def test_provider_rejects_ambiguous_duplicate_identity(monkeypatch):
    history = pd.DataFrame(
        {
            "competition": ["SA"],
            "season_start": [2020],
            "kickoff_utc": pd.to_datetime(["2020-09-19"], utc=True),
            "kickoff_time_available": [False],
            "home_team": ["Fiorentina"],
            "away_team": ["Torino"],
            "home_goals": [1],
            "away_goals": [0],
        },
        index=[20],
    )
    commits = [{"sha": "dup", "commit": {"committer": {"date": "2020-09-21T00:00:00Z"}}}]
    snapshot = """Date,Team 1,FT,HT,Team 2
Sat Sep 19 2020,Fiorentina,1-0,0-0,Torino
Sat Sep 19 2020,Fiorentina,1-0,0-0,Torino
"""
    monkeypatch.setattr(provider, "_commits", lambda *args, **kwargs: commits)
    monkeypatch.setattr(provider, "_snapshot", lambda *args, **kwargs: snapshot)
    assert provider.apply_footballcsv_italy_pit(history, cache_dir="/tmp/unused").empty


def test_source_rows_normalizes_identity():
    raw = """Date,Team 1,FT,HT,Team 2
Sat Sep 19 2020,Fiorentina,1-0,0-0,Torino
"""
    counts = provider._source_rows(raw)
    assert counts[("2020-09-19", "fiorentina", "torino", 1, 0)] == 1
