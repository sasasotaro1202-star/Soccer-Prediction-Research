from src.research.competition_taxonomy import classify_competition_kind, classify_stage


def test_competition_kind_covers_non_league_formats():
    assert classify_competition_kind("FA Cup") == "domestic_cup"
    assert classify_competition_kind("J.League Cup") == "league_cup"
    assert classify_competition_kind("UEFA Super Cup") == "super_cup"
    assert classify_competition_kind("Copa Libertadores") == "continental_club"
    assert classify_competition_kind("FIFA World Cup") == "international_tournament"
    assert classify_competition_kind("International Friendly") == "friendly"
    assert classify_competition_kind("Women's Super League") == "women"
    assert classify_competition_kind("U-20 World Cup") == "international_tournament"


def test_stage_taxonomy_separates_tournament_stage():
    assert classify_stage(round_name="Semi-finals") == "semi_final"
    assert classify_stage(round_name="Final") == "final"
    assert classify_stage(round_name="Promotion Playoff") == "playoff"
    assert classify_stage(round_name="Group A") == "group_stage"
    assert classify_stage(phase="Qualifying") == "qualifier"
