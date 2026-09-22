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


def test_country_commits_pages_until_oldest_required_bound(monkeypatch, tmp_path):
    import src.data.pit_openfootball_country as country

    calls = []
    pages = {
        1: [{"sha": "new", "commit": {"committer": {"date": "2024-01-01T00:00:00Z"}}}] * 100,
        2: [{"sha": "old", "commit": {"committer": {"date": "2020-01-01T00:00:00Z"}}}],
    }

    def fake_request(url, timeout=30):
        calls.append(url)
        page = int(url.split("page=")[-1])
        return pages[page]

    monkeypatch.setattr(country, "_request_json", fake_request)
    out = country._commits(
        "openfootball/england",
        "2020-21/1-premierleague.txt",
        str(tmp_path),
        min_commit_time="2020-06-01T00:00:00Z",
    )
    assert len(out) == 2
    assert any("page=2" in url for url in calls)
