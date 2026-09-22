import numpy as np
import pandas as pd

from src.research.engine import _primary_score_metrics_finite


def _rows():
    return {
        "score_logloss": [1.0],
        "exact_score_hit_rate": [0.1],
        "top3_score_hit_rate": [0.3],
        "top4_score_hit_rate": [0.4],
        "home_goals_mae": [1.0],
        "away_goals_mae": [1.0],
        "total_goals_mae": [1.2],
        "over_2_5_logloss": [0.7],
        "over_2_5_brier": [0.22],
        "btts_logloss": [0.68],
        "btts_brier": [0.21],
        "recency_score_logloss": [np.nan],
        "dc_score_logloss": [np.nan],
    }


def test_challenger_nan_does_not_block_primary_score_gate():
    assert _primary_score_metrics_finite(pd.DataFrame(_rows())) is True


def test_primary_nan_blocks_score_gate():
    rows = _rows()
    rows["score_logloss"] = [np.nan]
    assert _primary_score_metrics_finite(pd.DataFrame(rows)) is False
