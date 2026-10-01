import pandas as pd

from src.research.competition_target_diagnostics import build_competition_target_diagnostics


def test_competition_target_diagnostics_tracks_targets_and_leagues():
    rows = []
    for i in range(480):
        rows.append({
            "target": "O/U" if i % 4 < 2 else "BTTS",
            "block": i // 120,
            "match_id": f"m{i}",
            "competition": "EPL" if i % 2 == 0 else "LaLiga",
            "kickoff_utc": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=i),
            "prediction_probability": 0.72 if i % 3 else 0.28,
            "baseline_score_probability": 0.60 if i % 3 else 0.40,
            "actual": 1 if i % 3 else 0,
        })
    state = build_competition_target_diagnostics(pd.DataFrame(rows))
    assert state["production_usable"] is False
    assert state["status"] == "READY"
    assert len(state["target_competition_rows"]) == 4
    assert {r["target"] for r in state["target_competition_rows"]} == {"O/U", "BTTS"}
    assert {r["competition"] for r in state["target_competition_rows"]} == {"EPL", "LALIGA"}
    assert all(r["evidence_ready_for_specialist_research"] for r in state["target_competition_rows"])


def test_competition_target_diagnostics_rejects_duplicate_fixture_within_target():
    rows = [{
        "target": "O/U",
        "block": 0,
        "match_id": "same",
        "competition": "EPL",
        "kickoff_utc": pd.Timestamp("2024-01-01", tz="UTC"),
        "prediction_probability": 0.5,
        "baseline_score_probability": 0.5,
        "actual": 1,
    }]
    frame = pd.DataFrame(rows + [dict(rows[0])])
    try:
        build_competition_target_diagnostics(frame)
    except ValueError as exc:
        assert "duplicate fixture" in str(exc)
    else:
        raise AssertionError("duplicate fixture should fail closed")
