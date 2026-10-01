from src.data.matchday_intelligence_fetch import _sofascore_competition


def test_sofascore_maps_target_international_competitions():
    assert _sofascore_competition({"tournament": {"name": "FIFA World Cup"}}) == "WORLD_CUP"
    assert _sofascore_competition({"tournament": {"name": "AFC Asian Cup"}}) == "ASIAN_CUP"
    assert _sofascore_competition({"tournament": {"name": "International Friendly"}}) == "INTL_M"
    assert _sofascore_competition({"tournament": {"slug": "asian-games-women"}}) == "AG_W"
    assert _sofascore_competition({"tournament": {"slug": "international-friendly"}}) == "INTL_M"
