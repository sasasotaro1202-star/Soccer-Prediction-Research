import pandas as pd

from src.research.target_experience_learning import learn_target_specific_experience


def test_target_experience_learns_ou_and_btts_separately():
    rows = []
    for i in range(150):
        kickoff = pd.Timestamp("2021-01-01", tz="UTC") + pd.Timedelta(days=i)
        rows.append({
            "match_id": f"m{i}",
            "kickoff_utc": kickoff,
            "prediction_pit_cutoff_utc": kickoff - pd.Timedelta(hours=1),
            "prediction_pit_gate": "PASS",
            "experience_available_at_utc": kickoff + pd.Timedelta(hours=2),
            "actual_home_goals": 2 if i % 3 else 0,
            "actual_away_goals": 1 if i % 4 else 0,
            "over_2_5": 0.6 if i % 2 else 0.4,
            "btts_yes": 0.65 if i % 3 else 0.35,
        })
    state = learn_target_specific_experience(pd.DataFrame(rows))
    assert set(state["targets"]) == {"O/U", "BTTS"}
    assert all(v["production_usable"] is False for v in state["targets"].values())
    assert len(state["oos_rows"]) >= 6