from src.data.pit_engsoccerdata import _identity_key, _team_aliases, _snapshot_keys


def test_team_aliases_keep_only_unambiguous_aliases():
    text = """country,name,name_other,most_recent
England,Manchester United,Man United,
England,Manchester United,Man Utd,
England,Arsenal,Arsenal FC,
England,Other,Arsenal FC,
"""
    aliases = _team_aliases(text, "England")
    assert aliases["manunited"] == "manchesterunited"
    assert "arsenalfc" not in aliases


def test_snapshot_keys_and_source_keys_reconcile_aliases():
    aliases = {
        "manunited": "manchesterunited",
        "mancity": "manchestercity",
    }
    snapshot = """Date,Season,home,visitor,FT,hgoal,vgoal,tier
2021-09-25,2021,Manchester United,Manchester City,0-1,0,1,1
"""
    keys = _snapshot_keys(snapshot, "EPL", aliases=aliases)
    source_key = _identity_key(
        "2021-09-25",
        "Man United",
        "Man City",
        0,
        1,
        aliases=aliases,
    )
    assert source_key in keys


def test_alias_reconciliation_does_not_change_pit_time_rule():
    aliases = {"manunited": "manchesterunited"}
    source_key = _identity_key(
        "2022-09-24",
        "Man United",
        "Liverpool",
        2,
        1,
        aliases=aliases,
    )
    snapshot_key = _identity_key(
        "2022-09-24",
        "Manchester United",
        "Liverpool",
        2,
        1,
        aliases=aliases,
    )
    assert source_key == snapshot_key
