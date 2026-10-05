from datetime import datetime, timezone
import json

from src.research.prospective_inplay import (
    build_mature_rows,
    parse_live_snapshot,
)


def _event(state="in", clock=1800.0, home_score="1", away_score="0"):
    return {
        "id": "evt-1",
        "date": "2026-10-05T18:00:00Z",
        "competitions": [{
            "startDate": "2026-10-05T18:00:00Z",
            "status": {
                "clock": clock,
                "period": 1,
                "type": {"state": state},
            },
            "competitors": [
                {"id": "h", "homeAway": "home", "score": home_score,
                 "team": {"id": "h", "displayName": "Home FC"}},
                {"id": "a", "homeAway": "away", "score": away_score,
                 "team": {"id": "a", "displayName": "Away FC"}},
            ],
        }],
    }


def test_parse_live_snapshot_is_prospective_and_does_not_store_odds():
    summary = {
        "plays": [
            {"text": "Red Card", "team": {"id": "a"},
             "clock": {"value": 1200}, "period": {"number": 1}},
        ],
        "odds": [{"provider": {"name": "ignored"}}],
    }
    observed = datetime(2026, 10, 5, 18, 30, tzinfo=timezone.utc)
    row = parse_live_snapshot(
        event=_event(),
        summary=summary,
        league="eng.1",
        observed_at=observed,
        response_sha256="abc",
    )
    assert row is not None
    assert row["pit_verified"] is True
    assert row["source_available_at_utc"] == row["prediction_cutoff_utc"]
    assert row["retrieved_at_utc"] == row["prediction_cutoff_utc"]
    assert row["away_red_cards"] == 1
    assert "odds" not in row


def test_maturity_uses_later_observation_as_label_boundary():
    t0 = "2026-10-05T18:05:00+00:00"
    t1 = "2026-10-05T18:10:00+00:00"
    t2 = "2026-10-05T18:15:00+00:00"
    rows = [
        {
            "event_id": "evt-1",
            "status_state": "in",
            "observed_at_utc": t0,
            "kickoff_utc": "2026-10-05T18:00:00+00:00",
            "prediction_cutoff_utc": t0,
            "source_available_at_utc": t0,
            "retrieved_at_utc": t0,
            "pit_verified": True,
            "red_cards_known": True,
            "home_score": 0, "away_score": 0,
            "home_red_cards": 0, "away_red_cards": 0,
        },
        {
            "event_id": "evt-1",
            "status_state": "in",
            "observed_at_utc": t1,
            "kickoff_utc": "2026-10-05T18:00:00+00:00",
            "prediction_cutoff_utc": t1,
            "source_available_at_utc": t1,
            "retrieved_at_utc": t1,
            "pit_verified": True,
            "red_cards_known": True,
            "home_score": 1, "away_score": 0,
            "home_red_cards": 0, "away_red_cards": 0,
        },
        {
            "event_id": "evt-1",
            "status_state": "post",
            "observed_at_utc": t2,
            "kickoff_utc": "2026-10-05T18:00:00+00:00",
            "prediction_cutoff_utc": t2,
            "source_available_at_utc": t2,
            "retrieved_at_utc": t2,
            "pit_verified": True,
            "red_cards_known": True,
            "home_score": 1, "away_score": 0,
            "home_red_cards": 0, "away_red_cards": 0,
        },
    ]
    mature = build_mature_rows(rows, now=datetime(2026, 10, 5, 18, 20, tzinfo=timezone.utc))
    assert len(mature) == 2
    assert mature[0]["label_available_at_utc"] == t1
    assert mature[0]["next_event_type"] == "HOME_GOAL"
    assert mature[0]["final_home_goals"] == 1.0
    assert mature[0]["final_away_goals"] == 0.0


def test_ambiguous_multi_event_transition_is_excluded_without_imputation():
    base = {
        "event_id": "evt-2",
        "kickoff_utc": "2026-10-05T18:00:00+00:00",
        "prediction_cutoff_utc": "2026-10-05T18:05:00+00:00",
        "source_available_at_utc": "2026-10-05T18:05:00+00:00",
        "retrieved_at_utc": "2026-10-05T18:05:00+00:00",
        "pit_verified": True,
        "red_cards_known": True,
    }
    rows = [
        dict(base, status_state="in", observed_at_utc="2026-10-05T18:05:00+00:00",
             home_score=0, away_score=0, home_red_cards=0, away_red_cards=0),
        dict(base, status_state="in", observed_at_utc="2026-10-05T18:10:00+00:00",
             home_score=1, away_score=1, home_red_cards=0, away_red_cards=0),
        dict(base, status_state="post", observed_at_utc="2026-10-05T18:15:00+00:00",
             home_score=1, away_score=1, home_red_cards=0, away_red_cards=0),
    ]
    mature = build_mature_rows(rows)
    assert len(mature) == 1
    assert mature[0]["prediction_cutoff_utc"] == "2026-10-05T18:10:00+00:00"
    assert mature[0]["next_event_type"] == "NO_EVENT"
    assert mature[0]["next_event_time_utc"] is None
