from src.data.matchday_intelligence_fetch import parse_sofascore_event


def test_sofascore_qualification_uses_nested_unique_tournament():
    event = {
        "id": 12345,
        "startTimestamp": 1790366400,
        "tournament": {
            "name": "UEFA Champions League, Qualification",
            "slug": "uefa-champions-league-qualification",
            "uniqueTournament": {
                "name": "UEFA Champions League",
                "slug": "uefa-champions-league",
            },
        },
        "homeTeam": {"id": 1, "name": "Home FC"},
        "awayTeam": {"id": 2, "name": "Away FC"},
    }
    row = parse_sofascore_event(event)
    assert row is not None
    assert row["competition"] == "UCL"


def test_sofascore_top_level_unique_tournament_is_supported():
    event = {
        "id": 12346,
        "startTimestamp": 1790366400,
        "uniqueTournament": {
            "name": "Premier League",
            "slug": "premier-league",
        },
        "tournament": {"name": "England"},
        "homeTeam": {"id": 1, "name": "Home FC"},
        "awayTeam": {"id": 2, "name": "Away FC"},
    }
    row = parse_sofascore_event(event)
    assert row is not None
    assert row["competition"] == "EPL"
