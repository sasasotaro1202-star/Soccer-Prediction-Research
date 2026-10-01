from __future__ import annotations

import pandas as pd
import pytest

from src.data import versioned_result_pit as mod


def _history():
    return pd.DataFrame([
        {
            "match_id": "m1",
            "competition": "EPL",
            "season_start": 2024,
            "kickoff_utc": "2024-08-17T15:00:00Z",
            "kickoff_time_available": True,
            "home_team": "Arsenal",
            "away_team": "Wolves",
            "home_goals": 2.0,
            "away_goals": 0.0,
        },
        {
            "match_id": "m2",
            "competition": "EPL",
            "season_start": 2024,
            "kickoff_utc": "2024-08-18T15:00:00Z",
            "kickoff_time_available": True,
            "home_team": "Chelsea",
            "away_team": "Manchester City",
            "home_goals": 0.0,
            "away_goals": 2.0,
        },
    ])


def _snapshot():
    return {
        "matches": [
            {
                "date": "2024-08-17",
                "team1": "Arsenal FC",
                "team2": "Wolverhampton Wanderers FC",
                "score": {"ft": [2, 0]},
            },
            {
                "date": "2024-08-18",
                "team1": "Chelsea FC",
                "team2": "Manchester City FC",
                "score": {"ft": [0, 2]},
            },
        ]
    }


def test_versioned_bridge_requires_conservative_post_match_bound():
    kickoff = pd.Timestamp("2024-08-17T15:00:00Z")
    row = pd.Series({
        "kickoff_utc": kickoff,
        "kickoff_time_available": True,
    })
    bound, reason = mod._lower_bound(row)
    assert bound == kickoff + pd.Timedelta(minutes=180)
    assert reason == "kickoff_plus_180m"


def test_versioned_bridge_rejects_unknown_competitions():
    with pytest.raises(ValueError, match="unsupported competition"):
        mod.season_path("UCL", 2024)


def test_versioned_bridge_verifies_exact_result_once(monkeypatch, tmp_path):
    history = _history()
    commits = [
        {
            "sha": "early",
            "commit": {"committer": {"date": "2024-08-17T16:00:00Z"}},
        },
        {
            "sha": "late",
            "commit": {"committer": {"date": "2024-08-17T19:30:00Z"}},
        },
    ]

    def fake_commits(path, *, cache_dir, timeout):
        return commits

    def fake_file(path, sha, *, cache_dir, timeout):
        return _snapshot()

    monkeypatch.setattr(mod, "_commits", fake_commits)
    monkeypatch.setattr(mod, "_file_at_commit", fake_file)

    enriched, report = mod.apply_bulk(
        history,
        max_groups=1,
        rows_per_group=2,
        cache_dir=tmp_path,
    )

    assert enriched.loc[0, "versioned_result_evidence_status"] == "VERIFIED"
    assert enriched.loc[0, "versioned_result_commit_sha"] == "late"
    assert enriched.loc[0, "versioned_result_available_at_utc"] == "2024-08-17T19:30:00+00:00"
    assert int(report.loc[0, "verified_rows"]) == 1


def test_versioned_bridge_does_not_accept_ambiguous_snapshot_keys(monkeypatch, tmp_path):
    history = _history().iloc[[0]].copy()
    commits = [{
        "sha": "late",
        "commit": {"committer": {"date": "2024-08-17T19:30:00Z"}},
    }]
    ambiguous = {
        "matches": [
            {
                "date": "2024-08-17",
                "team1": "Arsenal FC",
                "team2": "Wolverhampton Wanderers FC",
                "score": {"ft": [2, 0]},
            },
            {
                "date": "2024-08-17",
                "team1": "Arsenal",
                "team2": "Wolverhampton Wanderers",
                "score": {"ft": [2, 0]},
            },
        ]
    }
    monkeypatch.setattr(mod, "_commits", lambda *args, **kwargs: commits)
    monkeypatch.setattr(mod, "_file_at_commit", lambda *args, **kwargs: ambiguous)

    enriched, report = mod.apply_bulk(
        history, max_groups=1, rows_per_group=1, cache_dir=tmp_path
    )

    assert enriched.loc[0, "versioned_result_evidence_status"] == "UNVERIFIABLE"
    assert int(report.loc[0, "ambiguous_match_keys"]) == 1


def test_versioned_bridge_is_bounded_by_group_limit(monkeypatch, tmp_path):
    history = pd.concat([
        _history(),
        _history().assign(
            match_id=["m3", "m4"],
            season_start=2023,
        ),
    ], ignore_index=True)
    seen = []

    monkeypatch.setattr(
        mod,
        "_commits",
        lambda path, **kwargs: seen.append(path) or [],
    )

    _, report = mod.apply_bulk(history, max_groups=1, cache_dir=tmp_path)
    assert len(report) == 1
    assert len(seen) == 1
