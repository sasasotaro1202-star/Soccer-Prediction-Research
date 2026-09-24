from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.matchday_effect import evaluate_matchday_effect, summarize_matchday_effect


def _predictions():
    return pd.DataFrame([
        {"match_id":"a","base_p_home":0.60,"base_p_draw":0.25,"base_p_away":0.15,"p_home":0.70,"p_draw":0.20,"p_away":0.10,"matchday_applied":True,"matchday_status":"APPLIED"},
        {"match_id":"b","base_p_home":0.20,"base_p_draw":0.30,"base_p_away":0.50,"p_home":0.20,"p_draw":0.30,"p_away":0.50,"matchday_applied":False,"matchday_status":"ABSENT"},
    ])


def test_matchday_effect_pairs_baseline_and_final_on_same_fixtures():
    result = evaluate_matchday_effect(
        _predictions(),
        pd.DataFrame([{"match_id":"a","target":0},{"match_id":"b","target":2}]),
    )
    assert result["n"] == 2
    assert result["changed_n"] == 1
    assert result["overall"]["delta"]["logloss"] < 0.0
    table = summarize_matchday_effect(result)
    assert set(table["match_id"]) == {"a","b"}
    assert table.loc[table["match_id"]=="a","matchday_prediction"].iloc[0] == 0


def test_matchday_effect_rejects_missing_baseline_probabilities():
    with pytest.raises(ValueError, match="missing"):
        evaluate_matchday_effect(
            _predictions().drop(columns=["base_p_home"]),
            pd.DataFrame([{"match_id":"a","target":0},{"match_id":"b","target":2}]),
        )
