import pandas as pd

from src.research.advanced_stats_ablation import _drop_advanced_features, _is_advanced_feature


def test_advanced_feature_filter_is_explicit_and_preserves_core_stats():
    frame = pd.DataFrame({
        "elo_diff": [10.0],
        "home_xg_avg_5": [1.5],
        "away_xg_avg_5": [0.8],
        "xg_diff_5": [0.7],
        "home_possession_avg_5": [55.0],
        "possession_diff_5": [10.0],
        "shots_diff_5": [2.0],
        "points_diff_5": [1.0],
    })
    out = _drop_advanced_features(frame)
    assert list(out.columns) == ["elo_diff", "shots_diff_5", "points_diff_5"]


def test_advanced_feature_token_detection():
    assert _is_advanced_feature("home_xg_avg_10")
    assert _is_advanced_feature("xg_diff_20")
    assert _is_advanced_feature("pass_accuracy_diff_5")
    assert not _is_advanced_feature("shots_diff_5")
