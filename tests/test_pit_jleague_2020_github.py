import pandas as pd

from src.data.pit_jleague_2020_github import (
    _source_rows,
    apply_jleague_2020_github_pit,
)


def _history():
    return pd.DataFrame(
        [
            {
                "competition": "J1",
                "season_start": 2020,
                "kickoff_utc": "2020-09-10T00:00:00Z",
                "home_team": "Shonan Bellmare",
                "away_team": "Urawa Red Diamonds",
                "home_goals": 2,
                "away_goals": 3,
                "kickoff_time_available": False,
                "pit_evidence_status": pd.NA,
            },
            {
                "competition": "J1",
                "season_start": 2020,
                "kickoff_utc": "2020-09-12T00:00:00Z",
                "home_team": "Kawasaki Frontale",
                "away_team": "Sagan Tosu",
                "home_goals": 3,
                "away_goals": 2,
                "kickoff_time_available": False,
                "pit_evidence_status": pd.NA,
            },
        ],
        index=[10, 11],
    )


def _csv(rows):
    return (
        "Year,Matchday,T1,T2,FTG_T1,FTG_T2,FR\n"
        + "".join(
            f"2020,{md},{home},{away},{hg},{ag},{fr}\n"
            for md, home, away, hg, ag, fr in rows
        )
    )


def test_source_rows_counts_exact_result_identities():
    text = _csv(
        [
            (1, "Shonan Bellmare", "Urawa Red Diamonds", 2, 3, "A"),
            (2, "Shonan Bellmare", "Urawa Red Diamonds", 2, 3, "A"),
        ]
    )
    counts = _source_rows(text)
    assert len(counts) == 1
    assert next(iter(counts.values())) == 2


def test_provider_accepts_only_snapshot_after_conservative_lower_bound(monkeypatch, tmp_path):
    commits = [
        {"sha": "before", "commit": {"committer": {"date": "2020-09-10T12:00:00Z"}}},
        {"sha": "after", "commit": {"committer": {"date": "2020-09-11T04:03:30Z"}}},
    ]
    snapshots = {
        "before": _csv([(1, "Shonan Bellmare", "Urawa Red Diamonds", 2, 3, "A")]),
        "after": _csv([(1, "Shonan Bellmare", "Urawa Red Diamonds", 2, 3, "A")]),
    }
    monkeypatch.setattr(
        "src.data.pit_jleague_2020_github._commits",
        lambda *args, **kwargs: commits,
    )
    monkeypatch.setattr(
        "src.data.pit_jleague_2020_github._snapshot_text",
        lambda sha, *args, **kwargs: snapshots[sha],
    )

    result = apply_jleague_2020_github_pit(_history(), cache_dir=str(tmp_path))

    assert result.index.tolist() == [10]
    assert result.loc[10, "pit_evidence_status"] == "VERIFIED"
    assert result.loc[10, "source_available_at_utc"] == "2020-09-11T04:03:30+00:00"
    assert "immutable_jleague_2020_snapshot" in result.loc[10, "pit_evidence_reason"]
    assert pd.isna(result.loc[11, "pit_evidence_status"]) is True


def test_provider_fails_closed_on_ambiguous_exact_identity(monkeypatch, tmp_path):
    commits = [
        {"sha": "after", "commit": {"committer": {"date": "2020-09-11T04:03:30Z"}}},
    ]
    snapshots = {
        "after": _csv(
            [
                (1, "Shonan Bellmare", "Urawa Red Diamonds", 2, 3, "A"),
                (15, "Shonan Bellmare", "Urawa Red Diamonds", 2, 3, "A"),
            ]
        )
    }
    monkeypatch.setattr(
        "src.data.pit_jleague_2020_github._commits",
        lambda *args, **kwargs: commits,
    )
    monkeypatch.setattr(
        "src.data.pit_jleague_2020_github._snapshot_text",
        lambda sha, *args, **kwargs: snapshots[sha],
    )

    result = apply_jleague_2020_github_pit(_history().iloc[[0]], cache_dir=str(tmp_path))

    assert result.empty
