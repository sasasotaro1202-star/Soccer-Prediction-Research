import pandas as pd
from src.research.target_specific_learning import run_target_specific_learning

def test_target_specific_learning_has_independent_targets_and_no_production_use():
    rows=[]
    for i in range(320):
        rows.append({
            "match_id": f"m{i}", "kickoff_utc": f"2020-01-{1+i:02d}" if i < 28 else f"2020-02-{(i-27):02d}",
            "home_team": f"H{i % 12}", "away_team": f"A{i % 12}",
            "home_goals": i % 4, "away_goals": (i*2) % 3, "pit_verified": True,
            "competition": "EPL", "feature_a": float(i % 7), "feature_b": float((i*3)%11),
        })
    df=pd.DataFrame(rows); df["kickoff_utc"]=pd.date_range("2020-01-01",periods=len(df),freq="D",tz="UTC")
    state=run_target_specific_learning(df,min_train=160,block_size=40,min_blocks=3)
    assert state["production_usable"] is False
    assert set(state["targets"]) == {"O/U","BTTS"}
    assert len(state["oos_rows"]) == 6
    assert {r["target"] for r in state["oos_rows"]} == {"O/U","BTTS"}
    assert all("selected_model" in r for r in state["oos_rows"])