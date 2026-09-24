import numpy as np
import pandas as pd
import pytest

from src.data.mom_player_history_adapter import (
    attach_mom_labels,
    build_mom_feature_rows,
    mom_data_contract_report,
)
from src.models.mom_model import MOM_FEATURE_COLUMNS


def _frames():
    base = pd.Timestamp("2025-01-01T12:00:00Z")
    fixtures = pd.DataFrame([
        {"id": 1, "date_utc": base, "home_team_id": 10, "away_team_id": 20, "goals_home": 2, "goals_away": 1, "is_played": True},
        {"id": 2, "date_utc": base + pd.Timedelta(days=3), "home_team_id": 20, "away_team_id": 10, "goals_home": 0, "goals_away": 1, "is_played": True},
        {"id": 3, "date_utc": base + pd.Timedelta(days=6), "home_team_id": 10, "away_team_id": 20, "goals_home": 1, "goals_away": 0, "is_played": True},
        {"id": 4, "date_utc": base + pd.Timedelta(days=9), "home_team_id": 20, "away_team_id": 10, "goals_home": 0, "goals_away": 2, "is_played": True},
        {"id": 5, "date_utc": base + pd.Timedelta(days=12), "home_team_id": 10, "away_team_id": 20, "goals_home": 2, "goals_away": 2, "is_played": True},
    ])
    player_rows = []
    player_ids = [101, 102, 103, 104, 105]
    for i, fid in enumerate(range(1, 6)):
        kickoff = base + pd.Timedelta(days=3 * (fid - 1))
        known_at = kickoff + pd.Timedelta(minutes=105)
        for j, pid in enumerate(player_ids):
            player_rows.append({
                "fixture_id": fid,
                "team_id": 10 if (fid + j) % 2 else 20,
                "player_id": pid + (0 if (fid + j) % 2 else 100),
                "player_name": f"P{pid}_{fid}",
                "is_starter": j < 4,
                "position": "F" if j < 2 else "M",
                "minutes": 90 if j < 4 else 20,
                "rating": 7.0 + 0.05 * j,
                "goals_total": 1 if j == 0 and fid % 2 else 0,
                "goals_assists": 1 if j == 1 and fid % 2 else 0,
                "shots_total": 2 + j,
                "passes_key": 1 + j,
            })
        # The test data intentionally alternates team identity so each target has
        # prior appearances for both teams.
        player_rows[-1]["team_id"] = 20 if player_rows[-1]["team_id"] == 10 else 10
    players = pd.DataFrame(player_rows)
    stats = pd.DataFrame([
        {"fixture_id": fid, "known_at": (base + pd.Timedelta(days=3 * (fid - 1))) + pd.Timedelta(minutes=105)}
        for fid in range(1, 6)
    ])
    player_stats = players[["fixture_id", "player_id"]].copy()
    player_stats["games_minutes"] = players["minutes"]
    player_stats["games_rating"] = players["rating"]
    player_stats["goals_total"] = players["goals_total"]
    player_stats["goals_assists"] = players["goals_assists"]
    player_stats["shots_total"] = players["shots_total"]
    player_stats["passes_key"] = players["passes_key"]
    return fixtures, players, player_stats, stats


def test_build_mom_features_uses_only_prior_known_player_facts():
    fixtures, players, player_stats, stats = _frames()
    out = build_mom_feature_rows(fixtures, players, player_stats, stats, min_history_appearances=1, lookback_appearances=5)
    assert not out.empty
    assert set(MOM_FEATURE_COLUMNS).issubset(out.columns)
    assert (out["feature_available_at_utc"] < out["kickoff_utc"]).all()
    assert out["pit_verified"].all()
    assert np.isfinite(out[list(MOM_FEATURE_COLUMNS)].to_numpy(dtype=float)).all()


def test_target_fixture_statistics_do_not_enter_candidate_features():
    fixtures, players, player_stats, stats = _frames()
    target_id = 5
    target_kickoff = fixtures.loc[fixtures["id"] == target_id, "date_utc"].iloc[0]
    players.loc[players["fixture_id"] == target_id, ["rating", "goals_total", "goals_assists", "shots_total", "passes_key"]] = 99
    out = build_mom_feature_rows(fixtures, players, player_stats, stats, min_history_appearances=1)
    target = out.loc[out["match_id"] == str(target_id)]
    assert not target.empty
    assert (target["feature_available_at_utc"] < target_kickoff).all()
    assert (target["recent_rating_ewm"] < 10).all()
    assert (target["recent_shots_per90_ewm"] < 10).all()


def test_label_attachment_never_converts_unmatched_target_to_negative():
    fixtures, players, stats = _frames()
    features = build_mom_feature_rows(fixtures, players, player_stats, stats, min_history_appearances=1)
    labels = pd.DataFrame([{"match_id": "5.0", "player_id": str(features.loc[features["match_id"] == "5.0", "player_id"].iloc[0])}])
    labelled = attach_mom_labels(features, labels)
    assert "is_motm" in labelled.columns
    assert labelled.groupby("match_id")["is_motm"].sum().eq(1).all()


def test_multiple_mom_winners_are_rejected():
    features = pd.DataFrame([{"match_id": "1", "player_id": "a"}])
    labels = pd.DataFrame([
        {"match_id": "1", "player_id": "a"},
        {"match_id": "1", "player_id": "b"},
    ])
    with pytest.raises(ValueError, match="multiple winners"):
        attach_mom_labels(features, labels)


def test_contract_reports_deferred_without_candidates():
    report = mom_data_contract_report(pd.DataFrame())
    assert report["status"] == "DEFERRED_NO_PIT_PLAYER_DATA"
    assert report["matches"] == 0
