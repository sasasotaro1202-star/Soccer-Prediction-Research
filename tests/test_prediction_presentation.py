import pytest

from src.prediction.presentation import format_prediction_row


def _complete_row():
    return {
        "match_id":"m1","kickoff_utc":"2026-01-01T00:00:00Z","home_team":"A","away_team":"B","model_version":"v1",
        "p_home":0.6,"p_draw":0.25,"p_away":0.15,
        "score_1":"1-0","score_1_probability":0.20,"score_2":"2-0","score_2_probability":0.15,"score_3":"1-1","score_3_probability":0.10,
        "mom_1_player_id":"p1","mom_1_probability":0.25,"mom_2_player_id":"p2","mom_2_probability":0.20,"mom_3_player_id":"p3","mom_3_probability":0.15,"mom_4_player_id":"p4","mom_4_probability":0.10,
        "market_over_0_5":0.9,"market_under_0_5":0.1,"market_over_1_5":0.7,"market_under_1_5":0.3,
        "market_over_2_5":0.45,"market_under_2_5":0.55,"market_over_3_5":0.25,"market_under_3_5":0.75,
        "market_over_4_5":0.1,"market_under_4_5":0.9,"market_btts_yes":0.4,"market_btts_no":0.6,
    }


def test_format_prediction_row_has_fixed_percentage_contract():
    out = format_prediction_row(_complete_row())
    assert out["1x2"] == {"home":"60.0%","draw":"25.0%","away":"15.0%"}
    assert len(out["score_top3"]) == 3
    assert len(out["mom_top4"]) == 4
    assert out["markets"]["over_2_5"] == "45.0%"


def test_format_prediction_row_fails_when_required_secondary_output_is_missing():
    row = _complete_row()
    row.pop("mom_4_player_id")
    with pytest.raises(ValueError, match="MOM TOP4 rank 4"):
        format_prediction_row(row)
