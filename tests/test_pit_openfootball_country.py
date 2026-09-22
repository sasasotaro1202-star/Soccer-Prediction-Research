import pandas as pd

from src.data.pit_openfootball_country import SOURCE_CONFIG, _norm, _row_key


def test_country_source_config_is_scope_limited():
    assert set(SOURCE_CONFIG) == {"EPL", "BL1", "LL", "SA", "ERE", "FL1"}
    assert all("{season}" in item["file"] for item in SOURCE_CONFIG.values())


def test_country_team_normalization_is_presentation_only():
    assert _norm("FC Köln") == "fckoln"
    assert _norm("Bayern München") == "bayernmunchen"


def test_country_row_key_uses_exact_result_identity():
    row = pd.Series({
        "kickoff_utc": "2024-08-17T15:00:00Z",
        "home_team": "Arsenal FC",
        "away_team": "Wolverhampton Wanderers FC",
        "home_goals": 2,
        "away_goals": 0,
    })
    assert _row_key(row) == (
        "2024-08-17",
        "arsenalfc",
        "wolverhamptonwanderersfc",
        2,
        0,
    )


def test_country_provider_records_empty_history_as_unverifiable(tmp_path, monkeypatch):
    import src.data.pit_openfootball_country as country

    history = pd.DataFrame([{
        "competition": "EPL",
        "season_start": 2024,
        "kickoff_utc": "2024-08-17T15:00:00Z",
        "kickoff_time_available": True,
        "home_team": "Arsenal FC",
        "away_team": "Wolverhampton Wanderers FC",
        "home_goals": 2,
        "away_goals": 0,
    }])
    monkeypatch.setattr(country, "_commits", lambda *args, **kwargs: [])
    out = country.apply_country_openfootball_pit(history, cache_dir=str(tmp_path))
    assert out.loc[0, "pit_evidence_status"] == "UNVERIFIABLE"
    assert out.loc[0, "pit_evidence_reason"] == "immutable_openfootball_commit_history_empty"
