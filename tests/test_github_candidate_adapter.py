from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.data.github_candidate_adapter import (
    adapt_candidate_csv,
    competition_path_matches,
    infer_match_columns,
)


def test_candidate_column_inference_handles_common_names():
    cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    mapped = infer_match_columns(cols)
    assert mapped["date"] == "Date"
    assert mapped["home_team"] == "HomeTeam"
    assert mapped["away_team"] == "AwayTeam"
    assert mapped["home_goals"] == "FTHG"
    assert mapped["away_goals"] == "FTAG"
    assert mapped["result"] == "FTR"


def test_candidate_path_must_identify_target_when_competition_column_is_missing():
    assert competition_path_matches("EPL", "data/premier_league_results.csv")
    assert not competition_path_matches("EPL", "data/bundesliga_results.csv")


def test_candidate_adapter_records_conservative_git_availability_time():
    raw = (
        "date,home_team,away_team,home_goals,away_goals,result\n"
        "2025-08-10,A,B,2,1,H\n"
        "2025-08-17,C,D,0,0,D\n"
    ).encode()
    available = datetime(2025, 9, 1, tzinfo=timezone.utc)
    out = adapt_candidate_csv(
        raw,
        competition="EPL",
        source_repo="example/repo",
        source_path="data/epl_results.csv",
        source_available_at_utc=available,
    )
    assert len(out) == 2
    assert out["competition"].eq("EPL").all()
    assert out["source_name"].eq("GitHub:example/repo").all()
    assert pd.to_datetime(out["source_available_at_utc"], utc=True).eq(pd.Timestamp(available)).all()
    assert out["match_id"].str.startswith("gh:").all()
    assert set(out["result"].tolist()) == {"H", "D"}
