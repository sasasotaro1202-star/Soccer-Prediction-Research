from __future__ import annotations

import pandas as pd

from src.data.competition_sources import TARGET_COMPETITIONS
from src.data.live_scope_coverage import active_target_coverage


def test_active_target_coverage_reports_every_target_even_when_unobserved():
    frame = pd.DataFrame([
        {"competition": "EPL", "matchday_source": "espn_scoreboard"},
        {"competition": "UCL", "matchday_source": "sofascore"},
        {"competition": "NOT_TARGET", "matchday_source": "other"},
    ])
    result = active_target_coverage(frame)
    assert result["active_target_count"] == len(TARGET_COMPETITIONS)
    assert set(result["per_competition"]) == set(TARGET_COMPETITIONS)
    assert result["per_competition"]["EPL"]["upcoming_rows"] == 1
    assert result["per_competition"]["EPL"]["sources"] == ["espn_scoreboard"]
    assert result["per_competition"]["UCL"]["upcoming_rows"] == 1
    assert "NOT_TARGET" not in result["per_competition"]
    assert result["targets_with_upcoming_fixtures"] == 2
    assert result["targets_without_upcoming_fixtures"] == len(TARGET_COMPETITIONS) - 2


def test_active_target_coverage_empty_snapshot_is_explicitly_all_unobserved():
    result = active_target_coverage(pd.DataFrame(columns=["competition", "matchday_source"]))
    assert result["active_target_count"] == len(TARGET_COMPETITIONS)
    assert result["targets_with_upcoming_fixtures"] == 0
    assert result["targets_without_upcoming_fixtures"] == len(TARGET_COMPETITIONS)
    assert result["unobserved_targets"] == list(TARGET_COMPETITIONS)
    assert result["upcoming_fixture_presence_pct"] == 0.0
