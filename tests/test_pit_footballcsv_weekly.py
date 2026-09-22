from unittest.mock import patch

import pandas as pd

from src.data.pit_footballcsv_weekly import _source_rows, apply_footballcsv_weekly_pit


CSV = """Date,Team 1,FT,HT,Team 2
Sat Sep 12 2020,Fulham,0-3,0-1,Arsenal
Sat Sep 19 2020,Arsenal,2-1,2-1,West Ham
"""


def _history():
    return pd.DataFrame(
        {
            "competition": ["EPL", "EPL"],
            "season_start": [2020, 2020],
            "kickoff_utc": pd.to_datetime(
                ["2020-09-12T14:00:00Z", "2020-09-19T14:00:00Z"], utc=True
            ),
            "kickoff_time_available": [True, True],
            "home_team": ["Fulham", "Arsenal"],
            "away_team": ["Arsenal", "West Ham"],
            "home_goals": [0, 2],
            "away_goals": [3, 1],
            "result": ["A", "H"],
            "pit_evidence_status": [pd.NA, pd.NA],
        },
        index=[10, 11],
    )


def test_source_rows_counts_exact_identity():
    counts = _source_rows(CSV + "Sat Sep 26 2020,Fulham,0-3,0-1,Arsenal\n")
    assert counts[("2020-09-12", "fulham", "arsenal", 0, 3)] == 1
    assert counts[("2020-09-26", "fulham", "arsenal", 0, 3)] == 1


def test_provider_accepts_only_snapshot_at_or_after_lower_bound(tmp_path):
    commits = [
        {"sha": "before", "commit": {"committer": {"date": "2020-09-12T15:00:00Z"}}},
        {"sha": "after", "commit": {"committer": {"date": "2020-09-12T17:30:00Z"}}},
    ]

    def fake_snapshot(path, sha, cache_dir, timeout):
        return CSV

    with patch(
        "src.data.pit_footballcsv_weekly._commits",
        return_value=commits,
    ), patch(
        "src.data.pit_footballcsv_weekly._snapshot",
        side_effect=fake_snapshot,
    ):
        result = apply_footballcsv_weekly_pit(_history().loc[[10]], cache_dir=str(tmp_path))

    assert result.loc[10, "pit_evidence_status"] == "VERIFIED"
    assert result.loc[10, "source_available_at_utc"] == "2020-09-12T17:30:00+00:00"


def test_provider_fails_closed_on_ambiguous_identity(tmp_path):
    ambiguous = CSV + "Sat Sep 12 2020,Fulham,0-3,0-1,Arsenal\n"
    commits = [
        {"sha": "after", "commit": {"committer": {"date": "2020-09-13T00:00:00Z"}}},
    ]

    with patch(
        "src.data.pit_footballcsv_weekly._commits",
        return_value=commits,
    ), patch(
        "src.data.pit_footballcsv_weekly._snapshot",
        return_value=ambiguous,
    ):
        result = apply_footballcsv_weekly_pit(_history().loc[[10]], cache_dir=str(tmp_path))

    assert result.empty


def test_provider_supports_multiple_competitions_from_explicit_config(tmp_path):
    assert set(__import__("src.data.pit_footballcsv_weekly", fromlist=["CONFIG"]).CONFIG) == {
        "EPL", "BL1", "LL", "FL1", "SA", "ERE"
    }
