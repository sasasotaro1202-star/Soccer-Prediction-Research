import numpy as np
import pandas as pd

from src.research.engine import _primary_score_metrics_finite


def _primary_rows():
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


def test_primary_score_gate_ignores_unavailable_challenger_metrics():
    assert _primary_score_metrics_finite(pd.DataFrame(_primary_rows())) is True


def test_primary_score_gate_rejects_nonfinite_primary_metric():
    rows = _primary_rows()
    rows["score_logloss"] = [np.nan]
    assert _primary_score_metrics_finite(pd.DataFrame(rows)) is False
