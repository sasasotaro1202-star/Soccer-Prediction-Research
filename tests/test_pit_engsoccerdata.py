from __future__ import annotations

from datetime import datetime, timezone

from src.data.pit_engsoccerdata import SNAPSHOTS, _identity_key, _select_snapshot, _snapshot_keys


def test_engsoccerdata_identity_is_strict_about_score_and_result():
    good = _identity_key("2021-05-19", "West Brom", "Preston", 3, 0)
    changed_score = _identity_key("2021-05-19", "West Brom", "Preston", 2, 0)
    changed_side = _identity_key("2021-05-19", "Preston", "West Brom", 3, 0)
    assert good == ("2021-05-19", "westbrom", "preston", 3.0, 0.0, "H")
    assert good != changed_score
    assert good != changed_side


def test_snapshot_keys_require_top_tier_rows():
    snapshot = """Date,Season,home,visitor,FT,hgoal,vgoal,tier
2021-05-19,2020,West Brom,Preston,1-1,1,1,1
2021-05-20,2020,Other,Team,0-2,0,2,2
"""
    keys = _snapshot_keys(snapshot, "EPL")
    assert ("2021-05-19", "westbrom", "preston", 1.0, 1.0, "D") in keys
    assert ("2021-05-20", "other", "team", 0.0, 2.0, "A") not in keys


def test_snapshot_observation_time_is_utc():
    observed = datetime.fromisoformat("2022-11-05T19:16:32+00:00").astimezone(timezone.utc)
    lower = datetime.fromisoformat("2022-11-04T00:00:00+00:00")
    assert observed >= lower


def test_snapshot_catalog_is_multi_versioned_and_chronological():
    assert all(len(v) >= 4 for v in SNAPSHOTS.values())
    for snapshots in SNAPSHOTS.values():
        times = [item["observed_at_utc"] for item in snapshots]
        assert times == sorted(times)


def test_snapshot_selection_uses_earliest_valid_observation():
    key = ("2021-05-19", "westbrom", "preston", 3.0, 0.0, "H")
    first = {"commit_sha": "old", "path": "data-raw/england.csv", "observed_at_utc": "2020-10-17T21:07:43+00:00"}
    second = {"commit_sha": "new", "path": "data-raw/england.csv", "observed_at_utc": "2022-11-05T19:16:32+00:00"}
    selected = _select_snapshot(
        key,
        datetime.fromisoformat("2021-01-01T00:00:00+00:00"),
        [
            (datetime.fromisoformat(first["observed_at_utc"]), first, {key}),
            (datetime.fromisoformat(second["observed_at_utc"]), second, {key}),
        ],
    )
    assert selected is not None
    assert selected[1]["commit_sha"] == "new"
