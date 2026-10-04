from __future__ import annotations

import pandas as pd
import pytest

from src.research.feature_set_variants import (
    aggregate_fold_metrics,
    select_feature_set,
    select_development_winner,
    variant_catalog,
)


def _frame():
    cols = {
        "match_id": ["m1", "m2"],
        "target": [0, 2],
        "pit_verified": [True, True],
        "kickoff_utc": ["2026-01-01T10:00:00Z", "2026-01-02T10:00:00Z"],
        "elo_diff": [10.0, -20.0],
        "comp_elo_diff": [5.0, -10.0],
        "home_elo_expected": [0.52, 0.48],
        "dynamic_elo_diff": [15.0, -12.0],
        "dynamic_home_elo_expected": [0.54, 0.49],
        "home_rest_hours": [72.0, 48.0],
        "away_rest_hours": [48.0, 72.0],
        "rest_diff_hours": [24.0, -24.0],
        "home_points_5": [2.0, 1.0],
        "away_points_5": [1.0, 2.0],
        "points_diff_5": [1.0, -1.0],
        "home_xg_avg_5": [1.2, 0.9],
        "away_xg_avg_5": [0.9, 1.1],
        "xg_diff_5": [0.3, -0.2],
        "home_points_ewma_5": [2.2, 1.0],
        "away_points_ewma_5": [1.1, 2.1],
        "points_ewma_diff_5": [1.1, -1.1],
        "h2h_games_5": [3.0, 2.0],
        "h2h_home_win_rate_5": [0.67, 0.0],
        "home_history_support_n": [8.0, 9.0],
        "away_history_support_n": [9.0, 8.0],
        "feature_source_max_available_at_utc": ["2026-01-01T08:00:00Z"] * 2,
        "source_available_at_utc": ["2026-01-03T00:00:00Z"] * 2,
    }
    return pd.DataFrame(cols)


def test_feature_variants_are_deterministic_and_do_not_include_targets():
    frame = _frame()
    first, first_meta = select_feature_set(frame, "advanced_stats_5")
    second, second_meta = select_feature_set(frame, "advanced_stats_5")
    assert first == second
    assert first_meta["feature_set_id"] == second_meta["feature_set_id"]
    forbidden = {
        "match_id",
        "target",
        "pit_verified",
        "feature_source_max_available_at_utc",
        "source_available_at_utc",
    }
    assert forbidden.isdisjoint(first)


def test_broad_variant_catalog_contains_multiple_window_and_information_patterns():
    names = [row["variant"] for row in variant_catalog()]
    assert len(names) >= 20
    for expected in (
        "form_3_only",
        "form_10_only",
        "form_3_5",
        "form_5_10",
        "basic_3_10",
        "advanced_3_10",
        "xg_possession_dense",
        "difference_heavy",
    ):
        assert expected in names


def test_window_specific_variants_are_distinct_when_columns_exist():
    frame = _frame()
    # Add the windows needed to verify that the selector is genuinely window-specific.
    frame["home_points_3"] = [2.0, 1.0]
    frame["away_points_3"] = [1.0, 2.0]
    frame["points_diff_3"] = [1.0, -1.0]
    frame["home_points_10"] = [3.0, 2.0]
    frame["away_points_10"] = [2.0, 3.0]
    frame["points_diff_10"] = [1.0, -1.0]
    only3, _ = select_feature_set(frame, "form_3_only")
    only10, _ = select_feature_set(frame, "form_10_only")
    assert "home_points_3" in only3
    assert "home_points_10" not in only3
    assert "home_points_10" in only10
    assert "home_points_3" not in only10


def test_xg_possession_dense_keeps_target_signal_subset():
    frame = _frame()
    frame["home_possession_5"] = [55.0, 48.0]
    frame["away_possession_5"] = [45.0, 52.0]
    frame["possession_diff_5"] = [10.0, -4.0]
    frame["home_pass_accuracy_5"] = [88.0, 84.0]
    frame["away_pass_accuracy_5"] = [82.0, 86.0]
    frame["pass_accuracy_diff_5"] = [6.0, -2.0]
    frame["home_corners_5"] = [6.0, 4.0]
    frame["away_corners_5"] = [3.0, 5.0]
    dense, _ = select_feature_set(frame, "xg_possession_dense")
    assert "home_xg_avg_5" in dense
    assert "home_possession_5" in dense
    assert "home_pass_accuracy_5" in dense
    assert "home_corners_5" in dense


def test_difference_heavy_is_restricted_to_difference_representation():
    frame = _frame()
    cols, _ = select_feature_set(frame, "difference_heavy")
    assert "elo_diff" in cols
    assert "points_diff_5" in cols
    assert "home_points_5" not in cols
    assert "away_xg_avg_5" not in cols


def test_aggregate_fold_metrics_separates_locked_blocks():
    wf = pd.DataFrame(
        {
            "n": [100, 100, 200, 200],
            "logloss": [1.0, 0.9, 0.8, 0.7],
            "brier": [0.3, 0.29, 0.28, 0.27],
            "accuracy": [0.5, 0.55, 0.6, 0.65],
            "ece": [0.1, 0.08, 0.06, 0.05],
        }
    )
    summary = aggregate_fold_metrics(wf, locked_blocks=2)
    assert summary["development_n"] == 200
    assert summary["locked_n"] == 400
    assert summary["development_logloss"] == pytest.approx(0.95)
    assert summary["locked_logloss"] == pytest.approx(0.75)


def test_development_winner_never_selects_from_locked_metrics():
    frame = pd.DataFrame(
        {
            "variant": ["strength_only", "candidate"],
            "feature_set_id": ["a", "b"],
            "development_logloss": [1.0, 0.95],
            "development_brier": [0.25, 0.24],
            "development_accuracy": [0.5, 0.55],
            "development_ece": [0.05, 0.04],
            "oos_window_signature": ["sig", "sig"],
            "locked_logloss": [9.0, 0.1],
        }
    )
    winner = select_development_winner(frame, "strength_only")
    assert winner["variant"] == "candidate"
    assert winner["locked_oos_used_for_selection"] is False


def test_development_winner_rejects_mismatched_oos_windows():
    frame = pd.DataFrame(
        {
            "variant": ["strength_only", "candidate"],
            "feature_set_id": ["a", "b"],
            "development_logloss": [1.0, 0.95],
            "development_brier": [0.25, 0.24],
            "development_accuracy": [0.5, 0.55],
            "development_ece": [0.05, 0.04],
            "oos_window_signature": ["sig1", "sig2"],
        }
    )
    with pytest.raises(RuntimeError, match="different OOS window"):
        select_development_winner(frame, "strength_only")
