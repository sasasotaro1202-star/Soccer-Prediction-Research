import pandas as pd

import src.data.pit_footballcsv_espana as provider


def _history():
    return pd.DataFrame(
        {
            "competition": ["LL", "LL"],
            "season_start": [2020, 2020],
            "kickoff_utc": pd.to_datetime(
                ["2020-09-10", "2020-09-12"], utc=True
            ),
            "kickoff_time_available": [False, False],
            "home_team": ["Valencia CF", "Barcelona"],
            "away_team": ["Athletic Club Bilbao", "Atletico Madrid"],
            "home_goals": [2, 2],
            "away_goals": [1, 1],
        },
        index=[10, 11],
    )


def test_source_rows_counts_unique_result_identity():
    raw = """Date,Team 1,FT,HT,Team 2
Mon Sep 7 2020,Valencia CF,2-1,1-0,Athletic Club Bilbao
"""
    counts = provider._source_rows(raw)
    assert counts[("2020-09-07", "valenciacf", "athleticclubbilbao", 2, 1)] == 1


def test_provider_accepts_snapshot_only_after_result_lower_bound(monkeypatch):
    history = _history()
    commits = [
        {
            "sha": "before",
            "commit": {"committer": {"date": "2020-09-10T23:59:00Z"}},
        },
        {
            "sha": "after",
            "commit": {"committer": {"date": "2020-09-13T00:00:00Z"}},
        },
    ]
    snapshot = """Date,Team 1,FT,HT,Team 2
Thu Sep 10 2020,Valencia CF,2-1,1-0,Athletic Club Bilbao
Sat Sep 12 2020,Barcelona,2-1,1-0,Atletico Madrid
"""
    monkeypatch.setattr(provider, "_commits", lambda *args, **kwargs: commits)
    monkeypatch.setattr(provider, "_snapshot", lambda *args, **kwargs: snapshot)

    result = provider.apply_footballcsv_espana_pit(history, cache_dir="/tmp/unused")
    assert result.index.tolist() == [10, 11]
    assert result.loc[10, "pit_evidence_status"] == "VERIFIED"
    assert result.loc[11, "pit_evidence_status"] == "VERIFIED"
    assert result.loc[10, "source_available_at_utc"].startswith("2020-09-13")
    assert result.loc[11, "source_available_at_utc"].startswith("2020-09-13")


def test_provider_fails_closed_on_ambiguous_exact_identity(monkeypatch):
    history = _history().iloc[[0]].copy()
    commits = [
        {
            "sha": "ambiguous",
            "commit": {"committer": {"date": "2020-09-12T00:00:00Z"}},
        }
    ]
    snapshot = """Date,Team 1,FT,HT,Team 2
Thu Sep 10 2020,Valencia CF,2-1,1-0,Athletic Club Bilbao
Thu Sep 10 2020,Valencia CF,2-1,1-0,Athletic Club Bilbao
"""
    monkeypatch.setattr(provider, "_commits", lambda *args, **kwargs: commits)
    monkeypatch.setattr(provider, "_snapshot", lambda *args, **kwargs: snapshot)

    result = provider.apply_footballcsv_espana_pit(history, cache_dir="/tmp/unused")
    assert result.empty


def test_provider_is_schema_explicit_and_does_not_touch_other_competitions(monkeypatch):
    history = _history()
    history.loc[11, "competition"] = "SA"
    monkeypatch.setattr(
        provider,
        "_commits",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not be called")),
    )
    result = provider.apply_footballcsv_espana_pit(history, cache_dir="/tmp/unused")
    assert result.empty
