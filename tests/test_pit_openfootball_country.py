import pandas as pd

from src.data.pit_openfootball_country import SOURCE_CONFIG, _commits, _norm, _row_key


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



def test_country_commit_history_paginates_when_first_page_is_full(monkeypatch, tmp_path):
    calls = []

    def fake_request(url):
        calls.append(url)
        if "page=1" in url:
            return [{"sha": str(i)} for i in range(100)]
        return [{"sha": "older"}]

    monkeypatch.setattr(
        "src.data.pit_openfootball_country._request_json",
        fake_request,
    )
    commits = _commits("openfootball/europe", "netherlands/2024-25_nl1.txt", str(tmp_path))
    assert len(commits) == 101
    assert calls == [
        calls[0],
        calls[1],
    ]
    assert "page=1" in calls[0]
    assert "page=2" in calls[1]
