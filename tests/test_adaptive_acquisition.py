from __future__ import annotations

import pandas as pd

from src.data.adaptive_acquisition import (
    AcquisitionConfig,
    deduplicate_history,
    expanded_window,
    identify_data_deficits,
    rank_discovery_targets,
    select_preferred_sources,
)


def test_expanded_window_is_bounded_and_progressive():
    config = AcquisitionConfig(
        rounds=4,
        expand_back_years=3,
        expand_forward_years=2,
        minimum_start_year=1990,
        maximum_end_year=2026,
    )
    assert expanded_window(2010, 2025, 0, config) == (2010, 2025)
    assert expanded_window(2010, 2025, 1, config) == (2007, 2026)
    assert expanded_window(2010, 2025, 4, config) == (1998, 2026)


def test_deduplicate_history_preserves_cross_source_evidence():
    frame = pd.DataFrame(
        [
            {"competition": "EPL", "season_start": 2025, "kickoff_utc": "2025-01-01", "home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0, "source_name": "source-a"},
            {"competition": "EPL", "season_start": 2025, "kickoff_utc": "2025-01-01", "home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0, "source_name": "source-a"},
            {"competition": "EPL", "season_start": 2025, "kickoff_utc": "2025-01-01", "home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0, "source_name": "source-b"},
        ]
    )
    out = deduplicate_history(frame)
    assert len(out) == 2
    assert set(out["source_name"]) == {"source-a", "source-b"}


def test_preferred_source_is_selected_without_dropping_alternates():
    coverage = pd.DataFrame(
        [
            {"competition": "EPL", "source": "A", "rows": 100, "status": "AVAILABLE", "pit_capable": False},
            {"competition": "EPL", "source": "B", "rows": 100, "status": "AVAILABLE", "pit_capable": True},
            {"competition": "UCL", "source": "C", "rows": 0, "status": "UNAVAILABLE", "pit_capable": False},
        ]
    )
    out = select_preferred_sources(coverage)
    assert len(out) == len(coverage)
    assert out.loc[out["competition"].eq("EPL"), "preferred_source"].iloc[0] == "B"


def test_discovery_targets_prioritize_sparse_competitions():
    history = pd.DataFrame({"competition": ["EPL", "EPL", "UCL"]})
    targets = rank_discovery_targets(history, ("EPL", "UCL", "J1", "J2"), limit=3)
    assert targets == ["J1", "J2", "UCL"]


def test_data_deficits_keep_sparse_competitions_in_scope_and_require_pit():
    history = pd.DataFrame({
        "competition": ["EPL"] * 5 + ["J1"],
        "season_start": [2024, 2024, 2025, 2025, 2025, 2025],
        "pit_verified": [True, True, True, True, False, True],
    })
    deficits = identify_data_deficits(
        history,
        ("EPL", "J1", "UCL"),
        minimum_rows_per_competition=3,
        minimum_seasons_per_competition=2,
    )
    assert "EPL" not in deficits
    assert deficits["J1"]["seasons"] == 1
    assert deficits["UCL"]["rows"] == 0
    assert deficits["J1"]["count_basis"] == "PIT_VERIFIED_ONLY"


def test_write_state_persists_discovered_acquisition_report(tmp_path):
    from src.data.adaptive_acquisition import write_state

    path = tmp_path / "adaptive_state.json"
    report = {"status": "COMPLETED", "acquired_candidates": 2, "acquired_rows": 120}
    write_state(
        path,
        config=AcquisitionConfig(),
        rounds=[{"round": 1, "history_rows_total": 10}],
        discovery={"status": "COMPLETED"},
        discovered_acquisition=report,
    )
    import json
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["discovered_acquisition"] == report
