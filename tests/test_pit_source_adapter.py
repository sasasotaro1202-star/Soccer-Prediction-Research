import pandas as pd

from src.data.pit_source_adapter import (
    COMPETITION_ADAPTERS,
    FootballDataWaybackAdapter,
    SourceEvidence,
    competition_adapter_matrix,
    source_url,
)


def test_all_15_competitions_are_explicitly_classified():
    assert len(COMPETITION_ADAPTERS) == 15
    matrix = competition_adapter_matrix()
    assert len(matrix) == 15
    assert set(matrix["competition"]) == set(COMPETITION_ADAPTERS)


def test_source_url_uses_fixed_and_acquisition_mappings():
    assert source_url("E0", 2025).endswith("/2526/E0.csv")
    assert source_url("CH", 2025).endswith("/2526/E1.csv")
    assert source_url("EPL", 2025).endswith("/2526/E0.csv")
    assert source_url("CHA", 2025).endswith("/2526/E1.csv")


def test_unknown_competition_fails_closed():
    try:
        source_url("UCL", 2025)
    except ValueError as exc:
        assert "No PIT source mapping" in str(exc)
    else:
        raise AssertionError("unverified competition must not receive a source URL")


def test_row_key_contains_completed_result_identity():
    row = pd.Series({
        "home_team": "Team A",
        "away_team": "Team B",
        "kickoff_utc": "2025-09-01T18:00:00Z",
        "home_goals": 2,
        "away_goals": 1,
        "result": "H",
    })
    assert FootballDataWaybackAdapter._row_key(row) == ("2025-09-01", "Team A", "Team B", 2.0, 1.0, "H")


def test_source_evidence_never_invents_timestamp():
    evidence = SourceEvidence(None, "UNVERIFIABLE", reason="no_archive_snapshot_contains_completed_result")
    assert evidence.source_available_at_utc is None
