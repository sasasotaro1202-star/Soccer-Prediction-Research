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

def test_target_experience_deduplicates_multiple_prediction_states_per_fixture():
    rows = []
    for i in range(121):
        kickoff = pd.Timestamp("2021-01-01", tz="UTC") + pd.Timedelta(days=i)
        rows.append({
            "match_id": f"m{i}",
            "kickoff_utc": kickoff,
            "prediction_pit_cutoff_utc": kickoff - pd.Timedelta(hours=2),
            "prediction_pit_gate": "PASS",
            "experience_available_at_utc": kickoff + pd.Timedelta(hours=2),
            "actual_home_goals": 2 if i % 2 else 0,
            "actual_away_goals": 1,
            "over_2_5": 0.6 if i % 2 else 0.4,
            "btts_yes": 0.65 if i % 3 else 0.35,
        })
    duplicate = dict(rows[0])
    duplicate["prediction_pit_cutoff_utc"] = duplicate["kickoff_utc"] - pd.Timedelta(hours=1)
    rows.append(duplicate)
    state = learn_target_specific_experience(pd.DataFrame(rows))
    assert state["targets"]["O/U"]["rows"] == 121
    assert state["targets"]["BTTS"]["rows"] == 121


def test_target_experience_blocks_legacy_ledger_schema(tmp_path):
    from src.research.target_experience_learning import write_target_specific_experience
    ledger = tmp_path / "legacy.csv"
    out = tmp_path / "target_experience"
    pd.DataFrame([{"match_id": "legacy", "p_home": 0.5}]).to_csv(ledger, index=False)
    state = write_target_specific_experience(str(ledger), str(out))
    assert state["status"] == "BLOCKED_LEDGER_SCHEMA"
    assert state["production_usable"] is False
    assert "experience_available_at_utc" in state["missing_columns"]
    assert (out / "target_experience_status.json").is_file()

def test_target_experience_empty_ledger_stays_in_warmup():
    state = learn_target_specific_experience(pd.DataFrame())
    assert state["status"] == "INSUFFICIENT_EXPERIENCE"
    assert state["production_usable"] is False
    assert "missing_columns" not in state


def test_write_target_experience_empty_headerless_ledger_stays_in_warmup(tmp_path):
    from src.research.target_experience_learning import write_target_specific_experience

    ledger = tmp_path / "empty.csv"
    out = tmp_path / "target_experience"
    ledger.write_text("\n", encoding="utf-8")
    state = write_target_specific_experience(str(ledger), str(out))

    assert state["status"] == "INSUFFICIENT_EXPERIENCE"
    assert state["production_usable"] is False
    assert (out / "target_experience_status.json").is_file()
